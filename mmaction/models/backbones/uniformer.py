# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from mmcv.cnn.bricks import DropPath
from mmengine.logging import MMLogger
from mmengine.model import BaseModule, ModuleList
from mmengine.runner.checkpoint import _load_checkpoint
from mmengine.utils import to_2tuple

from mmaction.registry import MODELS

logger = MMLogger.get_current_instance()

MODEL_PATH = 'https://download.openmmlab.com/mmaction/v1.0/recognition'
_MODELS = {
    'uniformer_small_in1k':
    os.path.join(MODEL_PATH,
                 'uniformerv1/uniformer_small_in1k_20221219-fe0a7ae0.pth'),
    'uniformer_base_in1k':
    os.path.join(MODEL_PATH,
                 'uniformerv1/uniformer_base_in1k_20221219-82c01015.pth'),
}


def conv_3xnxn(inp: int,
               oup: int,
               kernel_size: int = 3,
               stride: int = 3,
               groups: int = 1):
    """3D convolution with kernel size of 3xnxn.

    Args:
        inp (int): Dimension of input features.
        oup (int): Dimension of output features.
        kernel_size (int): The spatial kernel size (i.e., n).
            Defaults to 3.
        stride (int): The spatial stride.
            Defaults to 3.
        groups (int): Group number of operated features.
            Defaults to 1.
    """
    return nn.Conv3d(
        inp,
        oup, (3, kernel_size, kernel_size), (2, stride, stride), (1, 0, 0),
        groups=groups)


def conv_1xnxn(inp: int,
               oup: int,
               kernel_size: int = 3,
               stride: int = 3,
               groups: int = 1):
    """3D convolution with kernel size of 1xnxn.

    Args:
        inp (int): Dimension of input features.
        oup (int): Dimension of output features.
        kernel_size (int): The spatial kernel size (i.e., n).
            Defaults to 3.
        stride (int): The spatial stride.
            Defaults to 3.
        groups (int): Group number of operated features.
            Defaults to 1.
    """
    return nn.Conv3d(
        inp,
        oup, (1, kernel_size, kernel_size), (1, stride, stride), (0, 0, 0),
        groups=groups)


def conv_1x1x1(inp: int, oup: int, groups: int = 1):
    """3D convolution with kernel size of 1x1x1.

    Args:
        inp (int): Dimension of input features.
        oup (int): Dimension of output features.
        groups (int): Group number of operated features.
            Defaults to 1.
    """
    return nn.Conv3d(inp, oup, (1, 1, 1), (1, 1, 1), (0, 0, 0), groups=groups)


def conv_3x3x3(inp: int, oup: int, groups: int = 1):
    """3D convolution with kernel size of 3x3x3.

    Args:
        inp (int): Dimension of input features.
        oup (int): Dimension of output features.
        groups (int): Group number of operated features.
            Defaults to 1.
    """
    return nn.Conv3d(inp, oup, (3, 3, 3), (1, 1, 1), (1, 1, 1), groups=groups)


def conv_5x5x5(inp: int, oup: int, groups: int = 1):
    """3D convolution with kernel size of 5x5x5.

    Args:
        inp (int): Dimension of input features.
        oup (int): Dimension of output features.
        groups (int): Group number of operated features.
            Defaults to 1.
    """
    return nn.Conv3d(inp, oup, (5, 5, 5), (1, 1, 1), (2, 2, 2), groups=groups)


def bn_3d(dim):
    """3D batch normalization.

    Args:
        dim (int): Dimension of input features.
    """
    return nn.BatchNorm3d(dim)


class Mlp(BaseModule):
    """Multilayer perceptron.

    Args:
        in_features (int): Number of input features.
        hidden_features (int): Number of hidden features.
            Defaults to None.
        out_features (int): Number of output features.
            Defaults to None.
        drop (float): Dropout rate. Defaults to 0.0.
        init_cfg (dict, optional): Config dict for initialization.
            Defaults to None.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
        drop: float = 0.,
        init_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(BaseModule):
    """Self-Attention.

    Args:
        dim (int): Number of input features.
        num_heads (int): Number of attention heads.
            Defaults to 8.
        qkv_bias (bool): If True, add a learnable bias to query, key, value.
            Defaults to True.
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        attn_drop (float): Attention dropout rate.
            Defaults to 0.0.
        proj_drop (float): Dropout rate.
            Defaults to 0.0.
        init_cfg (dict, optional): Config dict for initialization.
            Defaults to None.
        init_cfg (dict, optional): The config of weight initialization.
            Defaults to None.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qk_scale: float = None,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        init_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version,
        # can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[
            2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CMlp(BaseModule):
    """Multilayer perceptron via convolution.

    Args:
        in_features (int): Number of input features.
        hidden_features (int): Number of hidden features.
            Defaults to None.
        out_features (int): Number of output features.
            Defaults to None.
        drop (float): Dropout rate. Defaults to 0.0.
        init_cfg (dict, optional): Config dict for initialization.
            Defaults to None.
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        drop=0.,
        init_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = conv_1x1x1(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = conv_1x1x1(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CBlock(BaseModule):
    """Convolution Block.

    Args:
        dim (int): Number of input features.
        mlp_ratio (float): Ratio of mlp hidden dimension
            to embedding dimension. Defaults to 4.
        drop (float): Dropout rate.
            Defaults to 0.0.
        drop_paths (float): Stochastic depth rates.
            Defaults to 0.0.
        init_cfg (dict, optional): Config dict for initialization.
            Defaults to None.
    """

    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.,
        drop: float = 0.,
        drop_path: float = 0.,
        init_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        self.pos_embed = conv_3x3x3(dim, dim, groups=dim)
        self.norm1 = bn_3d(dim)
        self.conv1 = conv_1x1x1(dim, dim, 1)
        self.conv2 = conv_1x1x1(dim, dim, 1)
        self.attn = conv_5x5x5(dim, dim, groups=dim)
        # NOTE: drop path for stochastic depth,
        # we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = bn_3d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(
            in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pos_embed(x)
        x = x + self.drop_path(
            self.conv2(self.attn(self.conv1(self.norm1(x)))))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SABlock(BaseModule):
    """Self-Attention Block.

    Args:
        dim (int): Number of input features.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dimension
            to embedding dimension. Defaults to 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value.
            Defaults to True.
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        drop (float): Dropout rate. Defaults to 0.0.
        attn_drop (float): Attention dropout rate. Defaults to 0.0.
        drop_paths (float): Stochastic depth rates.
            Defaults to 0.0.
        init_cfg (dict, optional): Config dict for initialization.
            Defaults to None.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.,
        qkv_bias: bool = False,
        qk_scale: float = None,
        drop: float = 0.,
        attn_drop: float = 0.,
        drop_path: float = 0.,
        init_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        self.pos_embed = conv_3x3x3(dim, dim, groups=dim)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop)
        # NOTE: drop path for stochastic depth,
        # we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pos_embed(x)
        B, C, T, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.transpose(1, 2).reshape(B, C, T, H, W)
        return x


class SpeicalPatchEmbed(BaseModule):
    """Image to Patch Embedding.

    Add extra temporal downsampling via temporal kernel size of 3.

    Args:
        img_size (int): Number of input size.
            Defaults to 224.
        patch_size (int): Number of patch size.
            Defaults to 16.
        in_chans (int): Number of input features.
            Defaults to 3.
        embed_dim (int): Number of output features.
            Defaults to 768.
        init_cfg (dict, optional): Config dict for initialization.
            Defaults to None.
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        init_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (
            img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = conv_3xnxn(
            in_chans,
            embed_dim,
            kernel_size=patch_size[0],
            stride=patch_size[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        B, _, T, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.reshape(B, T, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        return x


class PatchEmbed(BaseModule):
    """Image to Patch Embedding.

    Args:
        img_size (int): Number of input size.
            Defaults to 224.
        patch_size (int): Number of patch size.
            Defaults to 16.
        in_chans (int): Number of input features.
            Defaults to 3.
        embed_dim (int): Number of output features.
            Defaults to 768.
        init_cfg (dict, optional): Config dict for initialization.
            Defaults to None.
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        init_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (
            img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = conv_1xnxn(
            in_chans,
            embed_dim,
            kernel_size=patch_size[0],
            stride=patch_size[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        B, _, T, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.reshape(B, T, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        return x


@MODELS.register_module()
class UniFormer(BaseModule):
    """UniFormer.

    A pytorch implement of: `UniFormer: Unified Transformer
    for Efficient Spatiotemporal Representation Learning
    <https://arxiv.org/abs/2201.04676>`

    Args:
        depth (List[int]): List of depth in each stage.
            Defaults to [5, 8, 20, 7].
        img_size (int): Number of input size.
            Defaults to 224.
        in_chans (int): Number of input features.
            Defaults to 3.
        head_dim (int): Dimension of attention head.
            Defaults to 64.
        embed_dim (List[int]): List of embedding dimension in each layer.
            Defaults to [64, 128, 320, 512].
        mlp_ratio (float): Ratio of mlp hidden dimension
            to embedding dimension. Defaults to 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value.
            Defaults to True.
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        drop_rate (float): Dropout rate. Defaults to 0.0.
        attn_drop_rate (float): Attention dropout rate. Defaults to 0.0.
        drop_path_rate (float): Stochastic depth rates.
            Defaults to 0.0.
        pretrained2d (bool): Whether to load pretrained from 2D model.
            Defaults to True.
        pretrained (str): Name of pretrained model.
            Defaults to None.
        init_cfg (dict or list[dict]): Initialization config dict. Defaults to
            ``[
            dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
            dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
            ]``.
    """

    def __init__(
        self,
        depth: List[int] = [5, 8, 20, 7],
        img_size: int = 224,
        in_chans: int = 3,
        embed_dim: List[int] = [64, 128, 320, 512],
        head_dim: int = 64,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        qk_scale: float = None,
        drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.,
        pretrained2d: bool = True,
        pretrained: Optional[str] = None,
        init_cfg: Optional[Union[Dict, List[Dict]]] = [
            dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
            dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
        ]
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        self.pretrained = pretrained
        self.pretrained2d = pretrained2d
        self.patch_embed1 = SpeicalPatchEmbed(
            img_size=img_size,
            patch_size=4,
            in_chans=in_chans,
            embed_dim=embed_dim[0])
        self.patch_embed2 = PatchEmbed(
            img_size=img_size // 4,
            patch_size=2,
            in_chans=embed_dim[0],
            embed_dim=embed_dim[1])
        self.patch_embed3 = PatchEmbed(
            img_size=img_size // 8,
            patch_size=2,
            in_chans=embed_dim[1],
            embed_dim=embed_dim[2])
        self.patch_embed4 = PatchEmbed(
            img_size=img_size // 16,
            patch_size=2,
            in_chans=embed_dim[2],
            embed_dim=embed_dim[3])

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))
        ]  # stochastic depth decay rule
        num_heads = [dim // head_dim for dim in embed_dim]
        self.blocks1 = ModuleList([
            CBlock(
                dim=embed_dim[0],
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                drop_path=dpr[i]) for i in range(depth[0])
        ])
        self.blocks2 = ModuleList([
            CBlock(
                dim=embed_dim[1],
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                drop_path=dpr[i + depth[0]]) for i in range(depth[1])
        ])
        self.blocks3 = ModuleList([
            SABlock(
                dim=embed_dim[2],
                num_heads=num_heads[2],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i + depth[0] + depth[1]])
            for i in range(depth[2])
        ])
        self.blocks4 = ModuleList([
            SABlock(
                dim=embed_dim[3],
                num_heads=num_heads[3],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i + depth[0] + depth[1] + depth[2]])
            for i in range(depth[3])
        ])
        self.norm = bn_3d(embed_dim[-1])

    def _inflate_weight(self,
                        weight_2d: torch.Tensor,
                        time_dim: int,
                        center: bool = True) -> torch.Tensor:
        logger.info(f'Init center: {center}')
        if center:
            weight_3d = torch.zeros(*weight_2d.shape)
            weight_3d = weight_3d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
            middle_idx = time_dim // 2
            weight_3d[:, :, middle_idx, :, :] = weight_2d
        else:
            weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
            weight_3d = weight_3d / time_dim
        return weight_3d

    def _load_pretrained(self, pretrained: str = None) -> None:
        """Load ImageNet-1K pretrained model.

        The model is pretrained with ImageNet-1K.
        https://github.com/Sense-X/UniFormer

        Args:
            pretrained (str): Model name of ImageNet-1K pretrained model.
                Defaults to None.
        """
        if pretrained is not None:
            model_path = _MODELS[pretrained]
            logger.info(f'Load ImageNet pretrained model from {model_path}')
            state_dict = _load_checkpoint(model_path, map_location='cpu')
            state_dict_3d = self.state_dict()
            for k in state_dict.keys():
                if k in state_dict_3d.keys(
                ) and state_dict[k].shape != state_dict_3d[k].shape:
                    if len(state_dict_3d[k].shape) <= 2:
                        logger.info(f'Ignore: {k}')
                        continue
                    logger.info(f'Inflate: {k}, {state_dict[k].shape}' +
                                f' => {state_dict_3d[k].shape}')
                    time_dim = state_dict_3d[k].shape[2]
                    state_dict[k] = self._inflate_weight(
                        state_dict[k], time_dim)
            self.load_state_dict(state_dict, strict=False)

    def init_weights(self):
        """Initialize the weights in backbone."""
        if self.pretrained2d:
            logger = MMLogger.get_current_instance()
            logger.info(f'load model from: {self.pretrained}')
            self._load_pretrained(self.pretrained)
        else:
            if self.pretrained:
                self.init_cfg = dict(
                    type='Pretrained', checkpoint=self.pretrained)
            super().init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed1(x)
        x = self.pos_drop(x)
        for blk in self.blocks1:
            x = blk(x)
        x = self.patch_embed2(x)
        for blk in self.blocks2:
            x = blk(x)
        x = self.patch_embed3(x)
        for blk in self.blocks3:
            x = blk(x)
        x = self.patch_embed4(x)
        for blk in self.blocks4:
            x = blk(x)
        x = self.norm(x)
        return x
