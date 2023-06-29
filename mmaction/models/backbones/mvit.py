# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmcv.cnn.bricks import DropPath
from mmengine.logging import MMLogger
from mmengine.model import BaseModule, ModuleList
from mmengine.model.weight_init import trunc_normal_
from mmengine.runner.checkpoint import _load_checkpoint_with_prefix
from mmengine.utils import to_3tuple

from mmaction.registry import MODELS
from mmaction.utils import get_str_type
from ..utils.embed import PatchEmbed3D


def resize_pos_embed(pos_embed: torch.Tensor,
                     src_shape: Tuple[int],
                     dst_shape: Tuple[int],
                     mode: str = 'trilinear',
                     num_extra_tokens: int = 1) -> torch.Tensor:
    """Resize pos_embed weights.

    Args:
        pos_embed (torch.Tensor): Position embedding weights with shape
            [1, L, C].
        src_shape (tuple): The resolution of downsampled origin training
            image, in format (T, H, W).
        dst_shape (tuple): The resolution of downsampled new training
            image, in format (T, H, W).
        mode (str): Algorithm used for upsampling. Choose one from 'nearest',
            'linear', 'bilinear', 'bicubic' and 'trilinear'.
            Defaults to 'trilinear'.
        num_extra_tokens (int): The number of extra tokens, such as cls_token.
            Defaults to 1.

    Returns:
        torch.Tensor: The resized pos_embed of shape [1, L_new, C]
    """
    if src_shape[0] == dst_shape[0] and src_shape[1] == dst_shape[1] \
            and src_shape[2] == dst_shape[2]:
        return pos_embed
    assert pos_embed.ndim == 3, 'shape of pos_embed must be [1, L, C]'
    _, L, C = pos_embed.shape
    src_t, src_h, src_w = src_shape
    assert L == src_t * src_h * src_w + num_extra_tokens, \
        f"The length of `pos_embed` ({L}) doesn't match the expected " \
        f'shape ({src_t}*{src_h}*{src_w}+{num_extra_tokens}).' \
        'Please check the `img_size` argument.'
    extra_tokens = pos_embed[:, :num_extra_tokens]

    src_weight = pos_embed[:, num_extra_tokens:]
    src_weight = src_weight.reshape(1, src_t, src_h, src_w,
                                    C).permute(0, 4, 1, 2, 3)

    dst_weight = F.interpolate(
        src_weight, size=dst_shape, align_corners=False, mode=mode)
    dst_weight = torch.flatten(dst_weight, 2).transpose(1, 2)

    return torch.cat((extra_tokens, dst_weight), dim=1)


def resize_decomposed_rel_pos(rel_pos: torch.Tensor, q_size: int,
                              k_size: int) -> torch.Tensor:
    """Get relative positional embeddings according to the relative positions
    of query and key sizes.

    Args:
        rel_pos (Tensor): relative position embeddings (L, C).
        q_size (int): size of query q.
        k_size (int): size of key k.

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        resized = F.interpolate(
            # (L, C) -> (1, C, L)
            rel_pos.transpose(0, 1).unsqueeze(0),
            size=max_rel_dist,
            mode='linear',
        )
        # (1, C, L) -> (L, C)
        resized = resized.squeeze(0).transpose(0, 1)
    else:
        resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_h_ratio = max(k_size / q_size, 1.0)
    k_h_ratio = max(q_size / k_size, 1.0)
    q_coords = torch.arange(q_size)[:, None] * q_h_ratio
    k_coords = torch.arange(k_size)[None, :] * k_h_ratio
    relative_coords = (q_coords - k_coords) + (k_size - 1) * k_h_ratio

    return resized[relative_coords.long()]


def add_decomposed_rel_pos(attn: torch.Tensor,
                           q: torch.Tensor,
                           q_shape: Sequence[int],
                           k_shape: Sequence[int],
                           rel_pos_h: torch.Tensor,
                           rel_pos_w: torch.Tensor,
                           rel_pos_t: torch.Tensor,
                           with_cls_token: bool = False) -> torch.Tensor:
    """Spatiotemporal Relative Positional Embeddings."""
    sp_idx = 1 if with_cls_token else 0
    B, num_heads, _, C = q.shape
    q_t, q_h, q_w = q_shape
    k_t, k_h, k_w = k_shape

    Rt = resize_decomposed_rel_pos(rel_pos_t, q_t, k_t)
    Rh = resize_decomposed_rel_pos(rel_pos_h, q_h, k_h)
    Rw = resize_decomposed_rel_pos(rel_pos_w, q_w, k_w)

    r_q = q[:, :, sp_idx:].reshape(B, num_heads, q_t, q_h, q_w, C)
    rel_t = torch.einsum('bythwc,tkc->bythwk', r_q, Rt)
    rel_h = torch.einsum('bythwc,hkc->bythwk', r_q, Rh)
    rel_w = torch.einsum('bythwc,wkc->bythwk', r_q, Rw)
    rel_pos_embed = (
        rel_t[:, :, :, :, :, :, None, None] +
        rel_h[:, :, :, :, :, None, :, None] +
        rel_w[:, :, :, :, :, None, None, :])

    attn_map = attn[:, :, sp_idx:, sp_idx:].view(B, -1, q_t, q_h, q_w, k_t,
                                                 k_h, k_w)
    attn_map += rel_pos_embed
    attn[:, :, sp_idx:, sp_idx:] = attn_map.view(B, -1, q_t * q_h * q_w,
                                                 k_t * k_h * k_w)

    return attn


class MLP(BaseModule):
    """Two-layer multilayer perceptron.

    Comparing with :class:`mmcv.cnn.bricks.transformer.FFN`, this class allows
    different input and output channel numbers.

    Args:
        in_channels (int): The number of input channels.
        hidden_channels (int, optional): The number of hidden layer channels.
            If None, same as the ``in_channels``. Defaults to None.
        out_channels (int, optional): The number of output channels. If None,
            same as the ``in_channels``. Defaults to None.
        act_cfg (dict): The config of activation function.
            Defaults to ``dict(type='GELU')``.
        init_cfg (dict, optional): The config of weight initialization.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 hidden_channels: Optional[int] = None,
                 out_channels: Optional[int] = None,
                 act_cfg: Dict = dict(type='GELU'),
                 init_cfg: Optional[Union[Dict, List[Dict]]] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = build_activation_layer(act_cfg)
        self.fc2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


def attention_pool(x: torch.Tensor,
                   pool: nn.Module,
                   in_size: Tuple[int],
                   with_cls_token: bool = False,
                   norm: Optional[nn.Module] = None) -> tuple:
    """Pooling the feature tokens.

    Args:
        x (torch.Tensor): The input tensor, should be with shape
            ``(B, num_heads, L, C)`` or ``(B, L, C)``.
        pool (nn.Module): The pooling module.
        in_size (Tuple[int]): The shape of the input feature map.
        with_cls_token (bool): Whether concatenating class token into video
            tokens as transformer input. Defaults to True.
        norm (nn.Module, optional): The normalization module.
            Defaults to None.
    """
    ndim = x.ndim
    if ndim == 4:
        B, num_heads, L, C = x.shape
    elif ndim == 3:
        num_heads = 1
        B, L, C = x.shape
        x = x.unsqueeze(1)
    else:
        raise RuntimeError(f'Unsupported input dimension {x.shape}')

    T, H, W = in_size
    assert L == T * H * W + with_cls_token

    if with_cls_token:
        cls_tok, x = x[:, :, :1, :], x[:, :, 1:, :]

    # (B, num_heads, T*H*W, C) -> (B*num_heads, C, T, H, W)
    x = x.reshape(B * num_heads, T, H, W, C).permute(0, 4, 1, 2,
                                                     3).contiguous()
    x = pool(x)
    out_size = x.shape[2:]

    # (B*num_heads, C, T', H', W') -> (B, num_heads, T'*H'*W', C)
    x = x.reshape(B, num_heads, C, -1).transpose(2, 3)

    if with_cls_token:
        x = torch.cat((cls_tok, x), dim=2)

    if norm is not None:
        x = norm(x)

    if ndim == 3:
        x = x.squeeze(1)

    return x, out_size


class MultiScaleAttention(BaseModule):
    """Multiscale Multi-head Attention block.

    Args:
        in_dims (int): Number of input channels.
        out_dims (int): Number of output channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool): If True, add a learnable bias to query, key and
            value. Defaults to True.
        norm_cfg (dict): The config of normalization layers.
            Defaults to ``dict(type='LN')``.
        pool_kernel (tuple): kernel size for qkv pooling layers.
            Defaults to (3, 3, 3).
        stride_q (int): stride size for q pooling layer.
            Defaults to (1, 1, 1).
        stride_kv (int): stride size for kv pooling layer.
            Defaults to (1, 1, 1).
        rel_pos_embed (bool): Whether to enable the spatial and temporal
            relative position embedding. Defaults to True.
        residual_pooling (bool): Whether to enable the residual connection
            after attention pooling. Defaults to True.
        input_size (Tuple[int], optional): The input resolution, necessary
            if enable the ``rel_pos_embed``. Defaults to None.
        rel_pos_zero_init (bool): If True, zero initialize relative
            positional parameters. Defaults to False.
        with_cls_token (bool): Whether concatenating class token into video
            tokens as transformer input. Defaults to True.
        init_cfg (dict, optional): The config of weight initialization.
            Defaults to None.
    """

    def __init__(self,
                 in_dims: int,
                 out_dims: int,
                 num_heads: int,
                 qkv_bias: bool = True,
                 norm_cfg: Dict = dict(type='LN'),
                 pool_kernel: Tuple[int] = (3, 3, 3),
                 stride_q: Tuple[int] = (1, 1, 1),
                 stride_kv: Tuple[int] = (1, 1, 1),
                 rel_pos_embed: bool = True,
                 residual_pooling: bool = True,
                 input_size: Optional[Tuple[int]] = None,
                 rel_pos_zero_init: bool = False,
                 with_cls_token: bool = True,
                 init_cfg: Optional[dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.num_heads = num_heads
        self.with_cls_token = with_cls_token
        self.in_dims = in_dims
        self.out_dims = out_dims

        head_dim = out_dims // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(in_dims, out_dims * 3, bias=qkv_bias)
        self.proj = nn.Linear(out_dims, out_dims)

        # qkv pooling
        pool_padding = [k // 2 for k in pool_kernel]
        pool_dims = out_dims // num_heads

        def build_pooling(stride):
            pool = nn.Conv3d(
                pool_dims,
                pool_dims,
                pool_kernel,
                stride=stride,
                padding=pool_padding,
                groups=pool_dims,
                bias=False,
            )
            norm = build_norm_layer(norm_cfg, pool_dims)[1]
            return pool, norm

        self.pool_q, self.norm_q = build_pooling(stride_q)
        self.pool_k, self.norm_k = build_pooling(stride_kv)
        self.pool_v, self.norm_v = build_pooling(stride_kv)

        self.residual_pooling = residual_pooling

        self.rel_pos_embed = rel_pos_embed
        self.rel_pos_zero_init = rel_pos_zero_init
        if self.rel_pos_embed:
            # initialize relative positional embeddings
            assert input_size[1] == input_size[2]

            size = input_size[1]
            rel_dim = 2 * max(size // stride_q[1], size // stride_kv[1]) - 1
            self.rel_pos_h = nn.Parameter(torch.zeros(rel_dim, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(rel_dim, head_dim))
            self.rel_pos_t = nn.Parameter(
                torch.zeros(2 * input_size[0] - 1, head_dim))

    def init_weights(self) -> None:
        """Weight initialization."""
        super().init_weights()

        if (isinstance(self.init_cfg, dict)
                and get_str_type(self.init_cfg['type']) == 'Pretrained'):
            # Suppress rel_pos_zero_init if use pretrained model.
            return

        if not self.rel_pos_zero_init:
            trunc_normal_(self.rel_pos_h, std=0.02)
            trunc_normal_(self.rel_pos_w, std=0.02)
            trunc_normal_(self.rel_pos_t, std=0.02)

    def forward(self, x: torch.Tensor, in_size: Tuple[int]) -> tuple:
        """Forward the MultiScaleAttention."""
        B, N, _ = x.shape  # (B, H*W, C)

        # qkv: (B, H*W, 3, num_heads, C)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1)
        # q, k, v: (B, num_heads, H*W, C)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        q, q_shape = attention_pool(
            q,
            self.pool_q,
            in_size,
            norm=self.norm_q,
            with_cls_token=self.with_cls_token)
        k, k_shape = attention_pool(
            k,
            self.pool_k,
            in_size,
            norm=self.norm_k,
            with_cls_token=self.with_cls_token)
        v, v_shape = attention_pool(
            v,
            self.pool_v,
            in_size,
            norm=self.norm_v,
            with_cls_token=self.with_cls_token)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        if self.rel_pos_embed:
            attn = add_decomposed_rel_pos(attn, q, q_shape, k_shape,
                                          self.rel_pos_h, self.rel_pos_w,
                                          self.rel_pos_t, self.with_cls_token)

        attn = attn.softmax(dim=-1)
        x = attn @ v

        if self.residual_pooling:
            if self.with_cls_token:
                x[:, :, 1:, :] += q[:, :, 1:, :]
            else:
                x = x + q

        # (B, num_heads, H'*W', C'//num_heads) -> (B, H'*W', C')
        x = x.transpose(1, 2).reshape(B, -1, self.out_dims)
        x = self.proj(x)

        return x, q_shape


class MultiScaleBlock(BaseModule):
    """Multiscale Transformer blocks.

    Args:
        in_dims (int): Number of input channels.
        out_dims (int): Number of output channels.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of hidden dimensions in MLP layers.
            Defaults to 4.0.
        qkv_bias (bool): If True, add a learnable bias to query, key and
            value. Defaults to True.
        drop_path (float): Stochastic depth rate. Defaults to 0.
        norm_cfg (dict): The config of normalization layers.
            Defaults to ``dict(type='LN')``.
        act_cfg (dict): The config of activation function.
            Defaults to ``dict(type='GELU')``.
        qkv_pool_kernel (tuple): kernel size for qkv pooling layers.
            Defaults to (3, 3, 3).
        stride_q (int): stride size for q pooling layer.
            Defaults to (1, 1, 1).
        stride_kv (int): stride size for kv pooling layer.
            Defaults to (1, 1, 1).
        rel_pos_embed (bool): Whether to enable the spatial relative
            position embedding. Defaults to True.
        residual_pooling (bool): Whether to enable the residual connection
            after attention pooling. Defaults to True.
        with_cls_token (bool): Whether concatenating class token into video
            tokens as transformer input. Defaults to True.
        dim_mul_in_attention (bool): Whether to multiply the ``embed_dims`` in
            attention layers. If False, multiply it in MLP layers.
            Defaults to True.
        input_size (Tuple[int], optional): The input resolution, necessary
            if enable the ``rel_pos_embed``. Defaults to None.
        rel_pos_zero_init (bool): If True, zero initialize relative
            positional parameters. Defaults to False.
        init_cfg (dict, optional): The config of weight initialization.
            Defaults to None.
    """

    def __init__(
        self,
        in_dims: int,
        out_dims: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path: float = 0.0,
        norm_cfg: Dict = dict(type='LN'),
        act_cfg: Dict = dict(type='GELU'),
        qkv_pool_kernel: Tuple = (3, 3, 3),
        stride_q: Tuple = (1, 1, 1),
        stride_kv: Tuple = (1, 1, 1),
        rel_pos_embed: bool = True,
        residual_pooling: bool = True,
        with_cls_token: bool = True,
        dim_mul_in_attention: bool = True,
        input_size: Optional[Tuple[int]] = None,
        rel_pos_zero_init: bool = False,
        init_cfg: Optional[Dict] = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.with_cls_token = with_cls_token
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.norm1 = build_norm_layer(norm_cfg, in_dims)[1]
        self.dim_mul_in_attention = dim_mul_in_attention

        attn_dims = out_dims if dim_mul_in_attention else in_dims
        self.attn = MultiScaleAttention(
            in_dims,
            attn_dims,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            norm_cfg=norm_cfg,
            pool_kernel=qkv_pool_kernel,
            stride_q=stride_q,
            stride_kv=stride_kv,
            rel_pos_embed=rel_pos_embed,
            residual_pooling=residual_pooling,
            input_size=input_size,
            rel_pos_zero_init=rel_pos_zero_init,
            with_cls_token=with_cls_token)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = build_norm_layer(norm_cfg, attn_dims)[1]

        self.mlp = MLP(
            in_channels=attn_dims,
            hidden_channels=int(attn_dims * mlp_ratio),
            out_channels=out_dims,
            act_cfg=act_cfg)

        if in_dims != out_dims:
            self.proj = nn.Linear(in_dims, out_dims)
        else:
            self.proj = None

        if np.prod(stride_q) > 1:
            kernel_skip = [s + 1 if s > 1 else s for s in stride_q]
            padding_skip = [int(skip // 2) for skip in kernel_skip]
            self.pool_skip = nn.MaxPool3d(
                kernel_skip, stride_q, padding_skip, ceil_mode=False)

            if input_size is not None:
                input_size = to_3tuple(input_size)
                out_size = [size // s for size, s in zip(input_size, stride_q)]
                self.init_out_size = out_size
            else:
                self.init_out_size = None
        else:
            self.pool_skip = None
            self.init_out_size = input_size

    def forward(self, x: torch.Tensor, in_size: Tuple[int]) -> tuple:
        x_norm = self.norm1(x)
        x_attn, out_size = self.attn(x_norm, in_size)

        if self.dim_mul_in_attention and self.proj is not None:
            skip = self.proj(x_norm)
        else:
            skip = x

        if self.pool_skip is not None:
            skip, _ = attention_pool(
                skip,
                self.pool_skip,
                in_size,
                with_cls_token=self.with_cls_token)

        x = skip + self.drop_path(x_attn)
        x_norm = self.norm2(x)
        x_mlp = self.mlp(x_norm)

        if not self.dim_mul_in_attention and self.proj is not None:
            skip = self.proj(x_norm)
        else:
            skip = x

        x = skip + self.drop_path(x_mlp)

        return x, out_size


@MODELS.register_module()
class MViT(BaseModule):
    """Multi-scale ViT v2.

    A PyTorch implement of : `MViTv2: Improved Multiscale Vision Transformers
    for Classification and Detection <https://arxiv.org/abs/2112.01526>`_

    Inspiration from `the official implementation
    <https://github.com/facebookresearch/SlowFast>`_ and `the mmclassification
    implementation <https://github.com/open-mmlab/mmclassification>`_

    Args:
        arch (str | dict): MViT architecture. If use string, choose
            from 'tiny', 'small', 'base' and 'large'. If use dict, it should
            have below keys:

            - **embed_dims** (int): The dimensions of embedding.
            - **num_layers** (int): The number of layers.
            - **num_heads** (int): The number of heads in attention
              modules of the initial layer.
            - **downscale_indices** (List[int]): The layer indices to downscale
              the feature map.

            Defaults to 'base'.
        spatial_size (int): The expected input spatial_size shape.
            Defaults to 224.
        temporal_size (int): The expected input temporal_size shape.
            Defaults to 224.
        in_channels (int): The num of input channels. Defaults to 3.
        pretrained (str, optional): Name of pretrained model.
            Defaults to None.
        pretrained_type (str, optional): Type of pretrained model. choose from
            'imagenet', 'maskfeat', None. Defaults to None, which means load
            from same architecture.
        out_scales (int | Sequence[int]): The output scale indices.
            They should not exceed the length of ``downscale_indices``.
            Defaults to -1, which means the last scale.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.1.
        use_abs_pos_embed (bool): If True, add absolute position embedding to
            the patch embedding. Defaults to False.
        interpolate_mode (str): Select the interpolate mode for absolute
            position embedding vector resize. Defaults to "trilinear".
        pool_kernel (tuple): kernel size for qkv pooling layers.
            Defaults to (3, 3, 3).
        dim_mul (int): The magnification for ``embed_dims`` in the downscale
            layers. Defaults to 2.
        head_mul (int): The magnification for ``num_heads`` in the downscale
            layers. Defaults to 2.
        adaptive_kv_stride (int): The stride size for kv pooling in the initial
            layer. Defaults to (1, 8, 8).
        rel_pos_embed (bool): Whether to enable the spatial and temporal
            relative position embedding. Defaults to True.
        residual_pooling (bool): Whether to enable the residual connection
            after attention pooling. Defaults to True.
        dim_mul_in_attention (bool): Whether to multiply the ``embed_dims`` in
            attention layers. If False, multiply it in MLP layers.
            Defaults to True.
        with_cls_token (bool): Whether concatenating class token into video
            tokens as transformer input. Defaults to True.
        output_cls_token (bool): Whether output the cls_token. If set True,
            ``with_cls_token`` must be True. Defaults to True.
        rel_pos_zero_init (bool): If True, zero initialize relative
            positional parameters. Defaults to False.
        mlp_ratio (float): Ratio of hidden dimensions in MLP layers.
            Defaults to 4.0.
        qkv_bias (bool): enable bias for qkv if True. Defaults to True.
        norm_cfg (dict): Config dict for normalization layer for all output
            features. Defaults to ``dict(type='LN', eps=1e-6)``.
        patch_cfg (dict): Config dict for the patch embedding layer.
            Defaults to
            ``dict(kernel_size=(3, 7, 7),
                   stride=(2, 4, 4),
                   padding=(1, 3, 3))``.
        init_cfg (dict, optional): The Config for initialization. Defaults to
            ``[
            dict(type='TruncNormal', layer=['Conv2d', 'Conv3d'], std=0.02),
            dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
            dict(type='Constant', layer='LayerNorm', val=1., bias=0.02),
            ]``

    Examples:
        >>> import torch
        >>> from mmaction.registry import MODELS
        >>> from mmaction.utils import register_all_modules
        >>> register_all_modules()
        >>>
        >>> cfg = dict(type='MViT', arch='tiny', out_scales=[0, 1, 2, 3])
        >>> model = MODELS.build(cfg)
        >>> model.init_weights()
        >>> inputs = torch.rand(1, 3, 16, 224, 224)
        >>> outputs = model(inputs)
        >>> for i, output in enumerate(outputs):
        >>>     print(f'scale{i}: {output.shape}')
        scale0: torch.Size([1, 96, 8, 56, 56])
        scale1: torch.Size([1, 192, 8, 28, 28])
        scale2: torch.Size([1, 384, 8, 14, 14])
        scale3: torch.Size([1, 768, 8, 7, 7])
    """
    arch_zoo = {
        'tiny': {
            'embed_dims': 96,
            'num_layers': 10,
            'num_heads': 1,
            'downscale_indices': [1, 3, 8]
        },
        'small': {
            'embed_dims': 96,
            'num_layers': 16,
            'num_heads': 1,
            'downscale_indices': [1, 3, 14]
        },
        'base': {
            'embed_dims': 96,
            'num_layers': 24,
            'num_heads': 1,
            'downscale_indices': [2, 5, 21]
        },
        'large': {
            'embed_dims': 144,
            'num_layers': 48,
            'num_heads': 2,
            'downscale_indices': [2, 8, 44]
        },
    }
    num_extra_tokens = 1

    def __init__(
        self,
        arch: str = 'base',
        spatial_size: int = 224,
        temporal_size: int = 16,
        in_channels: int = 3,
        pretrained: Optional[str] = None,
        pretrained_type: Optional[str] = None,
        out_scales: Union[int, Sequence[int]] = -1,
        drop_path_rate: float = 0.,
        use_abs_pos_embed: bool = False,
        interpolate_mode: str = 'trilinear',
        pool_kernel: tuple = (3, 3, 3),
        dim_mul: int = 2,
        head_mul: int = 2,
        adaptive_kv_stride: tuple = (1, 8, 8),
        rel_pos_embed: bool = True,
        residual_pooling: bool = True,
        dim_mul_in_attention: bool = True,
        with_cls_token: bool = True,
        output_cls_token: bool = True,
        rel_pos_zero_init: bool = False,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        norm_cfg: Dict = dict(type='LN', eps=1e-6),
        patch_cfg: Dict = dict(
            kernel_size=(3, 7, 7), stride=(2, 4, 4), padding=(1, 3, 3)),
        init_cfg: Optional[Union[Dict, List[Dict]]] = [
            dict(type='TruncNormal', layer=['Conv2d', 'Conv3d'], std=0.02),
            dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.02),
            dict(type='Constant', layer='LayerNorm', val=1., bias=0.02),
        ]
    ) -> None:
        if pretrained:
            init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        super().__init__(init_cfg=init_cfg.copy())
        self.pretrained_type = pretrained_type

        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.arch_zoo), \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
        else:
            essential_keys = {
                'embed_dims', 'num_layers', 'num_heads', 'downscale_indices'
            }
            assert isinstance(arch, dict) and essential_keys <= set(arch), \
                f'Custom arch needs a dict with keys {essential_keys}'
            self.arch_settings = arch

        self.embed_dims = self.arch_settings['embed_dims']
        self.num_layers = self.arch_settings['num_layers']
        self.num_heads = self.arch_settings['num_heads']
        self.downscale_indices = self.arch_settings['downscale_indices']
        # Defaults take downscale_indices as downscale_indices
        self.dim_mul_indices = self.arch_settings.get(
            'dim_mul_indices', self.downscale_indices.copy())
        self.num_scales = len(self.downscale_indices) + 1
        self.stage_indices = {
            index - 1: i
            for i, index in enumerate(self.downscale_indices)
        }
        self.stage_indices[self.num_layers - 1] = self.num_scales - 1
        self.use_abs_pos_embed = use_abs_pos_embed
        self.interpolate_mode = interpolate_mode

        if isinstance(out_scales, int):
            out_scales = [out_scales]
        assert isinstance(out_scales, Sequence), \
            f'"out_scales" must by a sequence or int, ' \
            f'get {type(out_scales)} instead.'
        for i, index in enumerate(out_scales):
            if index < 0:
                out_scales[i] = self.num_scales + index
            assert 0 <= out_scales[i] <= self.num_scales, \
                f'Invalid out_scales {index}'
        self.out_scales = sorted(list(out_scales))

        # Set patch embedding
        _patch_cfg = dict(
            in_channels=in_channels,
            input_size=(temporal_size, spatial_size, spatial_size),
            embed_dims=self.embed_dims,
            conv_type='Conv3d',
        )
        _patch_cfg.update(patch_cfg)
        self.patch_embed = PatchEmbed3D(**_patch_cfg)
        self.patch_resolution = self.patch_embed.init_out_size

        # Set cls token
        if output_cls_token:
            assert with_cls_token is True, f'with_cls_token must be True if' \
                f'set output_cls_token to True, but got {with_cls_token}'
        self.with_cls_token = with_cls_token
        self.output_cls_token = output_cls_token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dims))

        # Set absolute position embedding
        if self.use_abs_pos_embed:
            num_patches = np.prod(self.patch_resolution)
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches + self.num_extra_tokens,
                            self.embed_dims))

        # stochastic depth decay rule
        dpr = np.linspace(0, drop_path_rate, self.num_layers)

        self.blocks = ModuleList()
        out_dims_list = [self.embed_dims]
        num_heads = self.num_heads
        stride_kv = adaptive_kv_stride
        input_size = self.patch_resolution
        for i in range(self.num_layers):
            if i in self.downscale_indices or i in self.dim_mul_indices:
                num_heads *= head_mul

            if i in self.downscale_indices:
                stride_q = [1, 2, 2]
                stride_kv = [max(s // 2, 1) for s in stride_kv]
            else:
                stride_q = [1, 1, 1]

            # Set output embed_dims
            if dim_mul_in_attention and i in self.dim_mul_indices:
                # multiply embed_dims in dim_mul_indices layers.
                out_dims = out_dims_list[-1] * dim_mul
            elif not dim_mul_in_attention and i + 1 in self.dim_mul_indices:
                # multiply embed_dims before dim_mul_indices layers.
                out_dims = out_dims_list[-1] * dim_mul
            else:
                out_dims = out_dims_list[-1]

            attention_block = MultiScaleBlock(
                in_dims=out_dims_list[-1],
                out_dims=out_dims,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
                norm_cfg=norm_cfg,
                qkv_pool_kernel=pool_kernel,
                stride_q=stride_q,
                stride_kv=stride_kv,
                rel_pos_embed=rel_pos_embed,
                residual_pooling=residual_pooling,
                with_cls_token=with_cls_token,
                dim_mul_in_attention=dim_mul_in_attention,
                input_size=input_size,
                rel_pos_zero_init=rel_pos_zero_init)
            self.blocks.append(attention_block)

            input_size = attention_block.init_out_size
            out_dims_list.append(out_dims)

            if i in self.stage_indices:
                stage_index = self.stage_indices[i]
                if stage_index in self.out_scales:
                    norm_layer = build_norm_layer(norm_cfg, out_dims)[1]
                    self.add_module(f'norm{stage_index}', norm_layer)

    def init_weights(self, pretrained: Optional[str] = None) -> None:
        # interpolate maskfeat relative position embedding
        if self.pretrained_type == 'maskfeat':
            logger = MMLogger.get_current_instance()
            pretrained = self.init_cfg['checkpoint']
            logger.info(f'load pretrained model from {pretrained}')
            state_dict = _load_checkpoint_with_prefix(
                'backbone.', pretrained, map_location='cpu')
            attn_rel_pos_keys = [
                k for k in state_dict.keys() if 'attn.rel_pos' in k
            ]
            for k in attn_rel_pos_keys:
                attn_rel_pos_pretrained = state_dict[k]
                attn_rel_pos_current = self.state_dict()[k]
                L1, dim1 = attn_rel_pos_pretrained.size()
                L2, dim2 = attn_rel_pos_current.size()
                if dim1 != dim2:
                    logger.warning(f'Dim mismatch in loading {k}, passing')
                else:
                    if L1 != L2:
                        interp_param = torch.nn.functional.interpolate(
                            attn_rel_pos_pretrained.t().unsqueeze(0),
                            size=L2,
                            mode='linear')
                        interp_param = \
                            interp_param.view(dim2, L2).permute(1, 0)
                        state_dict[k] = interp_param
                        logger.info(
                            f'{k} reshaped from {(L1, dim1)} to {L2, dim2}')
            msg = self.load_state_dict(state_dict, strict=False)
            logger.info(msg)

        elif self.pretrained_type is None:
            super().init_weights()

            if (isinstance(self.init_cfg, dict)
                    and get_str_type(self.init_cfg['type']) == 'Pretrained'):
                # Suppress default init if use pretrained model.
                return

        if self.use_abs_pos_embed:
            trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) ->\
            Tuple[Union[torch.Tensor, List[torch.Tensor]]]:
        """Forward the MViT."""
        B = x.shape[0]
        x, patch_resolution = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        if self.use_abs_pos_embed:
            x = x + resize_pos_embed(
                self.pos_embed,
                self.patch_resolution,
                patch_resolution,
                mode=self.interpolate_mode,
                num_extra_tokens=self.num_extra_tokens)

        if not self.with_cls_token:
            # Remove class token for transformer encoder input
            x = x[:, 1:]

        outs = []
        for i, block in enumerate(self.blocks):
            x, patch_resolution = block(x, patch_resolution)

            if i in self.stage_indices:
                stage_index = self.stage_indices[i]
                if stage_index in self.out_scales:
                    B, _, C = x.shape
                    x = getattr(self, f'norm{stage_index}')(x)
                    tokens = x.transpose(1, 2)
                    if self.with_cls_token:
                        patch_token = tokens[:, :, 1:].reshape(
                            B, C, *patch_resolution)
                        cls_token = tokens[:, :, 0]
                    else:
                        patch_token = tokens.reshape(B, C, *patch_resolution)
                        cls_token = None
                    if self.output_cls_token:
                        out = [patch_token, cls_token]
                    else:
                        out = patch_token
                    outs.append(out)

        return tuple(outs)
