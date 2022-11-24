# Copyright (c) OpenMMLab. All rights reserved.
from functools import lru_cache, reduce
from operator import mul
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from mmcv.cnn import build_activation_layer, build_conv_layer, build_norm_layer
from mmcv.cnn.bricks import DropPath
from mmengine.logging import MMLogger
from mmengine.model import BaseModule, ModuleList
from mmengine.model.weight_init import trunc_normal_
from mmengine.runner.checkpoint import _load_checkpoint

from mmaction.registry import MODELS


def window_partition(x: torch.Tensor,
                     window_size: Sequence[int]) -> torch.Tensor:
    """
    Args:
        x (torch.Tensor): The input features of shape :math:`(B, D, H, W, C)`.
        window_size (Sequence[int]): The window size, :math:`(w_d, w_h, w_w)`.

    Returns:
        torch.Tensor: The partitioned windows of shape
            :math:`(B*num_windows, w_d*w_h*w_w, C)`.
    """
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size[0], window_size[0], H // window_size[1],
               window_size[1], W // window_size[2], window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6,
                        7).contiguous().view(-1, reduce(mul, window_size), C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: Sequence[int], B: int,
                   D: int, H: int, W: int) -> torch.Tensor:
    """
    Args:
        windows (torch.Tensor): Input windows of shape
            :meth:`(B*num_windows, w_d, w_h, w_w, C)`.
        window_size (Sequence[int]): The window size, :meth:`(w_d, w_h, w_w)`.
        B (int): Batch size of feature maps.
        D (int): Temporal length of feature maps.
        H (int): Height of feature maps.
        W (int): Width of feature maps.

    Returns:
        torch.Tensor: The feature maps reversed from windows of
            shape :math:`(B, D, H, W, C)`.
    """
    x = windows.view(B, D // window_size[0], H // window_size[1],
                     W // window_size[2], window_size[0], window_size[1],
                     window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x


def get_window_size(
    x_size: Sequence[int],
    window_size: Sequence[int],
    shift_size: Optional[Sequence[int]] = None
) -> Union[Tuple[int], Tuple[Tuple[int]]]:
    """Calculate window size and shift size according to the input size.

    Args:
        x_size (Sequence[int]): The input size.
        window_size (Sequence[int]): The expected window size.
        shift_size (Sequence[int], optional): The expected shift size.
            Defaults to None.

    Returns:
        tuple: The calculated window size and shift size.
    """
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


# cache each stage results
@lru_cache()
def compute_mask(D: int, H: int, W: int, window_size: Sequence[int],
                 shift_size: Sequence[int],
                 device: Union[str, torch.device]) -> torch.Tensor:
    """Compute attention mask.

    Args:
        D (int): Temporal length of feature maps.
        H (int): Height of feature maps.
        W (int): Width of feature maps.
        window_size (Sequence[int]): The window size.
        shift_size (Sequence[int]): The shift size.
        device (str or :obj:`torch.device`): The device of the mask.

    Returns:
        torch.Tensor: The attention mask used for shifted window attention.
    """
    img_mask = torch.zeros((1, D, H, W, 1), device=device)  # 1 Dp Hp Wp 1
    cnt = 0
    for d in slice(-window_size[0]), slice(-window_size[0],
                                           -shift_size[0]), slice(
                                               -shift_size[0], None):
        for h in slice(-window_size[1]), slice(-window_size[1],
                                               -shift_size[1]), slice(
                                                   -shift_size[1], None):
            for w in slice(-window_size[2]), slice(-window_size[2],
                                                   -shift_size[2]), slice(
                                                       -shift_size[2], None):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1

    mask_windows = window_partition(img_mask,
                                    window_size)  # nW, ws[0]*ws[1]*ws[2], 1
    mask_windows = mask_windows.squeeze(-1)  # nW, ws[0]*ws[1]*ws[2]
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0,
                                      float(-100.0)).masked_fill(
                                          attn_mask == 0, float(0.0))
    return attn_mask


class WindowAttention3D(BaseModule):
    """Window based multi-head self attention (W-MSA) module with relative
    position bias. It supports both of shifted and non-shifted window.

    Args:
        embed_dims (int): Number of input channels.
        window_size (Sequence[int]): The temporal length, height and
            width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool):  If True, add a learnable bias to query,
            key, value. Defaults to True.
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        attn_drop (float): Dropout ratio of attention weight. Defaults to 0.0.
        proj_drop (float): Dropout ratio of output. Defaults to 0.0.
        init_cfg (dict, optional): Config dict for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims: int,
                 window_size: Sequence[int],
                 num_heads: int,
                 qkv_bias: bool = True,
                 qk_scale: Optional[float] = None,
                 attn_drop: float = 0.,
                 proj_drop: float = 0.,
                 init_cfg: Optional[Dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.embed_dims = embed_dims
        self.window_size = window_size  # Wd, Wh, Ww
        self.num_heads = num_heads
        head_dim = embed_dims // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # define a parameter table of relative position bias
        # # 2*Wd-1 * 2*Wh-1 * 2*Ww-1, nH
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) *
                        (2 * window_size[2] - 1), num_heads))

        # get pair-wise relative position index for
        # each token inside the window
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid(
            coords_d,
            coords_h,
            coords_w,
        ))  # 3, Wd, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
        # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = \
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        # shift to start from 0
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * \
                                    (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww
        self.register_buffer('relative_position_index',
                             relative_position_index)

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward function.

        Args:
            x (torch.Tensor): Input feature maps of shape
                :meth:`(B*num_windows, N, C)`.
            mask (torch.Tensor, optional): (0/-inf) mask of shape
                :meth:`(num_windows, N, N)`. Defaults to None.
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, nH, N, C

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index[:N, :N].reshape(-1)].reshape(
                N, N, -1)  # Wd*Wh*Ww,Wd*Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wd*Wh*Ww, Wd*Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)  # B_, nH, N, N

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(BaseModule):
    """Multilayer perceptron.

    Args:
        in_features (int): Number of input features.
        hidden_features (int, optional): Number of hidden features.
            Defaults to None.
        out_features (int, optional): Number of output features.
            Defaults to None.
        act_cfg (dict): Config dict for activation layer.
            Defaults to ``dict(type='GELU')``.
        drop (float): Dropout rate. Defaults to 0.0.
        init_cfg (dict, optional): Config dict for initialization.
            Defaults to None.
    """

    def __init__(self,
                 in_features: int,
                 hidden_features: Optional[int] = None,
                 out_features: Optional[int] = None,
                 act_cfg: Dict = dict(type='GELU'),
                 drop: float = 0.,
                 init_cfg: Optional[Dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = build_activation_layer(act_cfg)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwinTransformerBlock3D(BaseModule):
    """Swin Transformer Block.

    Args:
        embed_dims (int): Number of feature channels.
        num_heads (int): Number of attention heads.
        window_size (Sequence[int]): Window size. Defaults to ``(8, 7, 7)``.
        shift_size (Sequence[int]): Shift size for SW-MSA or W-MSA.
            Defaults to ``(0, 0, 0)``.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            Defaults to 4.0.
        qkv_bias (bool): If True, add a learnable bias to query, key, value.
            Defaults to True.
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        drop (float): Dropout rate. Defaults to 0.0.
        attn_drop (float): Attention dropout rate. Defaults to 0.0.
        drop_path (float): Stochastic depth rate. Defaults to 0.1.
        act_cfg (dict): Config dict for activation layer.
            Defaults to ``dict(type='GELU')``.
        norm_cfg (dict): Config dict for norm layer.
            Defaults to ``dict(type='LN')``.
        with_cp (bool): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Defaults to False.
        init_cfg (dict, optional): Config dict for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims: int,
                 num_heads: int,
                 window_size: Sequence[int] = (8, 7, 7),
                 shift_size: Sequence[int] = (0, 0, 0),
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = True,
                 qk_scale: Optional[float] = None,
                 drop: float = 0.,
                 attn_drop: float = 0.,
                 drop_path: float = 0.1,
                 act_cfg: Dict = dict(type='GELU'),
                 norm_cfg: Dict = dict(type='LN'),
                 with_cp: bool = False,
                 init_cfg: Optional[Dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.with_cp = with_cp

        assert 0 <= self.shift_size[0] < self.window_size[
            0], 'shift_size[0] must in [0, window_size[0])'
        assert 0 <= self.shift_size[1] < self.window_size[
            1], 'shift_size[1] must in [0, window_size[0])'
        assert 0 <= self.shift_size[2] < self.window_size[
            2], 'shift_size[2] must in [0, window_size[0])'

        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]

        _attn_cfg = {
            'embed_dims': embed_dims,
            'window_size': window_size,
            'num_heads': num_heads,
            'qkv_bias': qkv_bias,
            'qk_scale': qk_scale,
            'attn_drop': attn_drop,
            'proj_drop': drop
        }
        self.attn = WindowAttention3D(**_attn_cfg)

        self.drop_path = DropPath(drop_path) \
            if drop_path > 0. else nn.Identity()

        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]

        _mlp_cfg = {
            'in_features': embed_dims,
            'hidden_features': int(embed_dims * mlp_ratio),
            'act_cfg': act_cfg,
            'drop': drop
        }
        self.mlp = Mlp(**_mlp_cfg)

    def forward_part1(self, x: torch.Tensor,
                      mask_matrix: torch.Tensor) -> torch.Tensor:
        """Forward function part1."""
        B, D, H, W, C = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size,
                                                  self.shift_size)

        x = self.norm1(x)
        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, Dp, Hp, Wp, _ = x.shape
        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(
                x,
                shifts=(-shift_size[0], -shift_size[1], -shift_size[2]),
                dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        # partition windows
        x_windows = window_partition(shifted_x,
                                     window_size)  # B*nW, Wd*Wh*Ww, C
        # W-MSA/SW-MSA
        attn_windows = self.attn(
            x_windows, mask=attn_mask)  # B*nW, Wd*Wh*Ww, C
        # merge windows
        attn_windows = attn_windows.view(-1, *(window_size + (C, )))
        shifted_x = window_reverse(attn_windows, window_size, B, Dp, Hp,
                                   Wp)  # B D' H' W' C
        # reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = torch.roll(
                shifted_x,
                shifts=(shift_size[0], shift_size[1], shift_size[2]),
                dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :].contiguous()
        return x

    def forward_part2(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function part2."""
        return self.drop_path(self.mlp(self.norm2(x)))

    def forward(self, x: torch.Tensor,
                mask_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input features of shape :math:`(B, D, H, W, C)`.
            mask_matrix (torch.Tensor): Attention mask for cyclic shift.
        """

        shortcut = x
        if self.with_cp:
            x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix)
        else:
            x = self.forward_part1(x, mask_matrix)
        x = shortcut + self.drop_path(x)

        if self.with_cp:
            x = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x = x + self.forward_part2(x)

        return x


class PatchMerging(BaseModule):
    """Patch Merging Layer.

    Args:
        embed_dims (int): Number of input channels.
        norm_cfg (dict): Config dict for norm layer.
            Defaults to ``dict(type='LN')``.
        init_cfg (dict, optional): Config dict for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims: int,
                 norm_cfg: Dict = dict(type='LN'),
                 init_cfg: Optional[Dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.embed_dims = embed_dims
        self.mid_embed_dims = 4 * embed_dims
        self.out_embed_dims = 2 * embed_dims
        self.reduction = nn.Linear(
            self.mid_embed_dims, self.out_embed_dims, bias=False)
        self.norm = build_norm_layer(norm_cfg, self.mid_embed_dims)[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform patch merging.

        Args:
            x (torch.Tensor): Input feature maps of shape
                :math:`(B, D, H, W, C)`.

        Returns:
            torch.Tensor: The merged feature maps of shape
                :math:`(B, D, H/2, W/2, 2*C)`.
        """
        B, D, H, W, C = x.shape

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, :, 0::2, 0::2, :]  # B D H/2 W/2 C
        x1 = x[:, :, 1::2, 0::2, :]  # B D H/2 W/2 C
        x2 = x[:, :, 0::2, 1::2, :]  # B D H/2 W/2 C
        x3 = x[:, :, 1::2, 1::2, :]  # B D H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B D H/2 W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class BasicLayer(BaseModule):
    """A basic Swin Transformer layer for one stage.

    Args:
        embed_dims (int): Number of feature channels.
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (Sequence[int]): Local window size.
            Defaults to ``(8, 7, 7)``.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            Defaults to 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value.
            Defaults to True.
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        drop (float): Dropout rate. Defaults to 0.0.
        attn_drop (float): Attention dropout rate. Defaults to 0.0.
        drop_paths (float or Sequence[float]): Stochastic depth rates.
            Defaults to 0.0.
        act_cfg (dict): Config dict for activation layer.
            Defaults to ``dict(type='GELU')``.
        norm_cfg (dict, optional): Config dict for norm layer.
            Defaults to ``dict(type='LN')``.
        downsample (:class:`PatchMerging`, optional): Downsample layer
            at the end of the layer. Defaults to None.
        with_cp (bool): Use checkpoint or not. Using checkpoint will
            save some memory while slowing down the training speed.
            Defaults to False.
        init_cfg (dict, optional): Config dict for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims: int,
                 depth: int,
                 num_heads: int,
                 window_size: Sequence[int] = (8, 7, 7),
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = True,
                 qk_scale: Optional[float] = None,
                 drop: float = 0.,
                 attn_drop: float = 0.,
                 drop_paths: Union[float, Sequence[float]] = 0.,
                 act_cfg: Dict = dict(type='GELU'),
                 norm_cfg: Dict = dict(type='LN'),
                 downsample: Optional[PatchMerging] = None,
                 with_cp: bool = False,
                 init_cfg: Optional[Dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.embed_dims = embed_dims
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.depth = depth
        self.with_cp = with_cp

        if not isinstance(drop_paths, Sequence):
            drop_paths = [drop_paths] * depth

        # build blocks
        self.blocks = ModuleList()
        for i in range(depth):
            _block_cfg = {
                'embed_dims': embed_dims,
                'num_heads': num_heads,
                'window_size': window_size,
                'shift_size': (0, 0, 0) if (i % 2 == 0) else self.shift_size,
                'mlp_ratio': mlp_ratio,
                'qkv_bias': qkv_bias,
                'qk_scale': qk_scale,
                'drop': drop,
                'attn_drop': attn_drop,
                'drop_path': drop_paths[i],
                'act_cfg': act_cfg,
                'norm_cfg': norm_cfg,
                'with_cp': with_cp
            }

            block = SwinTransformerBlock3D(**_block_cfg)
            self.blocks.append(block)

        self.downsample = downsample
        if self.downsample is not None:
            self.downsample = downsample(
                embed_dims=embed_dims, norm_cfg=norm_cfg)

    def forward(self,
                x: torch.Tensor,
                do_downsample: bool = True) -> torch.Tensor:
        """Forward function.

        Args:
            x (torch.Tensor): Input feature maps of shape
                :math:`(B, C, D, H, W)`.
            do_downsample (bool): Whether to downsample the output of
                the current layer. Defaults to True.
        """
        # calculate attention mask for SW-MSA
        B, C, D, H, W = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size,
                                                  self.shift_size)
        x = rearrange(x, 'b c d h w -> b d h w c')
        Dp = int(np.ceil(D / window_size[0])) * window_size[0]
        Hp = int(np.ceil(H / window_size[1])) * window_size[1]
        Wp = int(np.ceil(W / window_size[2])) * window_size[2]
        attn_mask = compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device)
        for blk in self.blocks:
            x = blk(x, attn_mask)

        if self.downsample is not None and do_downsample:
            x = self.downsample(x)
        return x

    @property
    def out_embed_dims(self):
        if self.downsample is not None:
            return self.downsample.out_embed_dims
        else:
            return self.embed_dims


class PatchEmbed3D(BaseModule):
    """Video to Patch Embedding.

    Args:
        patch_size (Sequence[int] or int]): Patch token size.
            Defaults to ``(2, 4, 4)``.
        in_channels (int): Number of input video channels. Defaults to 3.
        embed_dims (int): Dimensions of embedding. Defaults to 96.
        conv_cfg: (dict): Config dict for convolution layer.
            Defaults to ``dict(type='Conv3d')``.
        norm_cfg (dict, optional): Config dict for norm layer.
            Defaults to None.
        init_cfg (dict, optional): Config dict for initialization.
            Defaults to None.
    """

    def __init__(self,
                 patch_size: Union[Sequence[int], int] = (2, 4, 4),
                 in_channels: int = 3,
                 embed_dims: int = 96,
                 norm_cfg: Optional[Dict] = None,
                 conv_cfg: Dict = dict(type='Conv3d'),
                 init_cfg: Optional[Dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dims = embed_dims

        self.proj = build_conv_layer(
            conv_cfg,
            in_channels,
            embed_dims,
            kernel_size=patch_size,
            stride=patch_size)

        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        else:
            self.norm = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform video to patch embedding.

        Args:
            x (torch.Tensor): The input videos of shape
                :math:`(B, C, D, H, W)`. In most cases, C is 3.

        Returns:
            torch.Tensor: The video patches of shape
                :math:`(B, embed_dims, Dp, Hp, Wp)`.
        """

        _, _, D, H, W = x.size()
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x,
                      (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0,
                          self.patch_size[0] - D % self.patch_size[0]))

        x = self.proj(x)  # B C Dp Wp Wp
        if self.norm is not None:
            Dp, Hp, Wp = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)  # B Dp*Hp*Wp C
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dims, Dp, Hp, Wp)

        return x


@MODELS.register_module()
class SwinTransformer3D(BaseModule):
    """Video Swin Transformer backbone.

    A pytorch implement of: `Video Swin Transformer
    <https://arxiv.org/abs/2106.13230>`_

    Args:
        arch (str or dict): Video Swin Transformer architecture. If use string,
            choose from 'tiny', 'small', 'base' and 'large'. If use dict, it
            should have below keys:
            - **embed_dims** (int): The dimensions of embedding.
            - **depths** (Sequence[int]): The number of blocks in each stage.
            - **num_heads** (Sequence[int]): The number of heads in attention
            modules of each stage.
        pretrained (str, optional): Name of pretrained model.
            Defaults to None.
        pretrained2d (bool): Whether to load pretrained 2D model.
            Defaults to True.
        patch_size (int or Sequence(int)): Patch size.
            Defaults to ``(2, 4, 4)``.
        in_channels (int): Number of input image channels. Defaults to 3.
        window_size (Sequence[int]): Window size. Defaults to ``(8, 7, 7)``.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            Defaults to 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value.
            Defaults to True.
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        drop_rate (float): Dropout rate. Defaults to 0.0.
        attn_drop_rate (float): Attention dropout rate. Defaults to 0.0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.1.
        act_cfg (dict): Config dict for activation layer.
            Defaults to ``dict(type='GELU')``.
        norm_cfg (dict): Config dict for norm layer.
            Defaults to ``dict(type='LN')``.
        patch_norm (bool): If True, add normalization after patch embedding.
            Defaults to True.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Defaults to -1.
        with_cp (bool): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Defaults to False.
        out_indices (Sequence[int]): Indices of output feature.
            Defaults to ``(3, )``.
        out_after_downsample (bool): Whether to output the feature map of a
            stage after the following downsample layer. Defaults to False.
        init_cfg (dict or list[dict]): Initialization config dict. Defaults to
            ``[
            dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
            dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
            ]``.
    """
    arch_zoo = {
        **dict.fromkeys(['t', 'tiny'],
                        {'embed_dims': 96,
                         'depths': [2, 2, 6, 2],
                         'num_heads': [3, 6, 12, 24]}),
        **dict.fromkeys(['s', 'small'],
                        {'embed_dims': 96,
                         'depths': [2, 2, 18, 2],
                         'num_heads': [3, 6, 12, 24]}),
        **dict.fromkeys(['b', 'base'],
                        {'embed_dims': 128,
                         'depths': [2, 2, 18, 2],
                         'num_heads': [4, 8, 16, 32]}),
        **dict.fromkeys(['l', 'large'],
                        {'embed_dims': 192,
                         'depths': [2, 2, 18, 2],
                         'num_heads': [6, 12, 24, 48]}),
    }  # yapf: disable

    def __init__(
        self,
        arch: Union[str, Dict],
        pretrained: Optional[str] = None,
        pretrained2d: bool = True,
        patch_size: Union[int, Sequence[int]] = (2, 4, 4),
        in_channels: int = 3,
        window_size: Sequence[int] = (8, 7, 7),
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.1,
        act_cfg: Dict = dict(type='GELU'),
        norm_cfg: Dict = dict(type='LN'),
        patch_norm: bool = True,
        frozen_stages: int = -1,
        with_cp: bool = False,
        out_indices: Sequence[int] = (3, ),
        out_after_downsample: bool = False,
        init_cfg: Optional[Union[Dict, List[Dict]]] = [
            dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
            dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
        ]
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        self.pretrained = pretrained
        self.pretrained2d = pretrained2d

        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.arch_zoo), \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
        else:
            essential_keys = {'embed_dims', 'depths', 'num_heads'}
            assert isinstance(arch, dict) and set(arch) == essential_keys, \
                f'Custom arch needs a dict with keys {essential_keys}'
            self.arch_settings = arch

        self.embed_dims = self.arch_settings['embed_dims']
        self.depths = self.arch_settings['depths']
        self.num_heads = self.arch_settings['num_heads']
        assert len(self.depths) == len(self.num_heads)
        self.num_layers = len(self.depths)
        assert 1 <= self.num_layers <= 4
        self.out_indices = out_indices
        assert max(out_indices) < self.num_layers
        self.out_after_downsample = out_after_downsample
        self.frozen_stages = frozen_stages
        self.window_size = window_size
        self.patch_size = patch_size

        _patch_cfg = {
            'patch_size': patch_size,
            'in_channels': in_channels,
            'embed_dims': self.embed_dims,
            'norm_cfg': norm_cfg if patch_norm else None,
            'conv_cfg': dict(type='Conv3d')
        }
        self.patch_embed = PatchEmbed3D(**_patch_cfg)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        total_depth = sum(self.depths)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]  # stochastic depth decay rule

        # build layers
        self.layers = ModuleList()
        embed_dims = [self.embed_dims]
        for i, (depth, num_heads) in \
                enumerate(zip(self.depths, self.num_heads)):
            downsample = PatchMerging if i < self.num_layers - 1 else None
            _layer_cfg = {
                'embed_dims': embed_dims[-1],
                'depth': depth,
                'num_heads': num_heads,
                'window_size': window_size,
                'mlp_ratio': mlp_ratio,
                'qkv_bias': qkv_bias,
                'qk_scale': qk_scale,
                'drop': drop_rate,
                'attn_drop': attn_drop_rate,
                'drop_paths': dpr[:depth],
                'act_cfg': act_cfg,
                'norm_cfg': norm_cfg,
                'downsample': downsample,
                'with_cp': with_cp
            }

            layer = BasicLayer(**_layer_cfg)
            self.layers.append(layer)

            dpr = dpr[depth:]
            embed_dims.append(layer.out_embed_dims)

        if self.out_after_downsample:
            self.num_features = embed_dims[1:]
        else:
            self.num_features = embed_dims[:-1]

        for i in out_indices:
            if norm_cfg is not None:
                norm_layer = build_norm_layer(norm_cfg,
                                              self.num_features[i])[1]
            else:
                norm_layer = nn.Identity()

            self.add_module(f'norm{i}', norm_layer)

        self._freeze_stages()

    def _freeze_stages(self) -> None:
        """Prevent all the parameters from being optimized before
        ``self.frozen_stages``."""
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def inflate_weights(self, logger: MMLogger) -> None:
        """Inflate the swin2d parameters to swin3d.

        The differences between swin3d and swin2d mainly lie in an extra
        axis. To utilize the pretrained parameters in 2d model, the weight
        of swin2d models should be inflated to fit in the shapes of the
        3d counterpart.

        Args:
            logger (MMLogger): The logger used to print debugging information.
        """
        checkpoint = _load_checkpoint(self.pretrained, map_location='cpu')
        state_dict = checkpoint['model']

        # delete relative_position_index since we always re-init it
        relative_position_index_keys = [
            k for k in state_dict.keys() if 'relative_position_index' in k
        ]
        for k in relative_position_index_keys:
            del state_dict[k]

        # delete attn_mask since we always re-init it
        attn_mask_keys = [k for k in state_dict.keys() if 'attn_mask' in k]
        for k in attn_mask_keys:
            del state_dict[k]
        state_dict['patch_embed.proj.weight'] = \
            state_dict['patch_embed.proj.weight'].unsqueeze(2).\
            repeat(1, 1, self.patch_size[0], 1, 1) / self.patch_size[0]

        # bicubic interpolate relative_position_bias_table if not match
        relative_position_bias_table_keys = [
            k for k in state_dict.keys() if 'relative_position_bias_table' in k
        ]
        for k in relative_position_bias_table_keys:
            relative_position_bias_table_pretrained = state_dict[k]
            relative_position_bias_table_current = self.state_dict()[k]
            L1, nH1 = relative_position_bias_table_pretrained.size()
            L2, nH2 = relative_position_bias_table_current.size()
            L2 = (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
            wd = self.window_size[0]
            if nH1 != nH2:
                logger.warning(f'Error in loading {k}, passing')
            else:
                if L1 != L2:
                    S1 = int(L1**0.5)
                    relative_position_bias_table_pretrained_resized = \
                        torch.nn.functional.interpolate(
                            relative_position_bias_table_pretrained.permute(
                                1, 0).view(1, nH1, S1, S1),
                            size=(2 * self.window_size[1] - 1,
                                  2 * self.window_size[2] - 1),
                            mode='bicubic')
                    relative_position_bias_table_pretrained = \
                        relative_position_bias_table_pretrained_resized. \
                        view(nH2, L2).permute(1, 0)
            state_dict[k] = relative_position_bias_table_pretrained.repeat(
                2 * wd - 1, 1)

        # In the original swin2d checkpoint, the last layer of the
        # backbone is the norm layer, and the original attribute
        # name is `norm`. We changed it to `norm3` which means it
        # is the last norm layer of stage 4.
        if hasattr(self, 'norm3'):
            state_dict['norm3.weight'] = state_dict['norm.weight']
            state_dict['norm3.bias'] = state_dict['norm.bias']
            del state_dict['norm.weight']
            del state_dict['norm.bias']

        msg = self.load_state_dict(state_dict, strict=False)
        logger.info(msg)

    def init_weights(self) -> None:
        """Initialize the weights in backbone."""
        if self.pretrained2d:
            logger = MMLogger.get_current_instance()
            logger.info(f'load model from: {self.pretrained}')
            # Inflate 2D model into 3D model.
            self.inflate_weights(logger)
        else:
            if self.pretrained:
                self.init_cfg = dict(
                    type='Pretrained', checkpoint=self.pretrained)
            super().init_weights()

    def forward(self, x: torch.Tensor) -> \
            Union[Tuple[torch.Tensor], torch.Tensor]:
        """Forward function for Swin3d Transformer."""
        x = self.patch_embed(x)

        x = self.pos_drop(x)

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x.contiguous(), do_downsample=self.out_after_downsample)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(x)
                out = rearrange(out, 'b d h w c -> b c d h w').contiguous()
                outs.append(out)

            if layer.downsample is not None and not self.out_after_downsample:
                x = layer.downsample(x)

            if i < self.num_layers - 1:
                x = rearrange(x, 'b d h w c -> b c d h w')

        if len(outs) == 1:
            return outs[0]

        return tuple(outs)

    def train(self, mode: bool = True) -> None:
        """Convert the model into training mode while keep layers frozen."""
        super(SwinTransformer3D, self).train(mode)
        self._freeze_stages()
