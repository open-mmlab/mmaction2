# Modified by Jialian Wu from https://github.com/facebookresearch/detectron2
# /blob/main/detectron2/modeling/backbone/vit.py
import logging
import math
from functools import partial

import fvcore.nn.weight_init as weight_init
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from detectron2.layers import CNNBlockBase, Conv2d, ShapeSpec, get_norm
from detectron2.modeling.backbone.backbone import Backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from timm.models.layers import DropPath, Mlp, trunc_normal_

from projects.videochat.models.centernet.modeling.backbone.fpn_p5 import \
    LastLevelP6P7_P5
from .utils import (PatchEmbed, add_decomposed_rel_pos, get_abs_pos,
                    window_partition, window_unpartition)

logger = logging.getLogger(__name__)

__all__ = ['ViT']


class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        input_size=None,
    ):
        """
        Args: dim (int): Number of input channels. num_heads (int): Number
        of attention heads. qkv_bias (bool:  If True, add a learnable bias
        to query, key, value. rel_pos (bool): If True, add relative
        positional embeddings to the attention map. rel_pos_zero_init (
        bool): If True, zero initialize relative positional parameters.
        input_size (int or None): Input resolution for calculating the
        relative positional parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(
                torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(
                torch.zeros(2 * input_size[1] - 1, head_dim))

            if not rel_pos_zero_init:
                trunc_normal_(self.rel_pos_h, std=0.02)
                trunc_normal_(self.rel_pos_w, std=0.02)

    def forward(self, x):
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads,
                                  -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h,
                                          self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W,
                            -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x


class ResBottleneckBlock(CNNBlockBase):
    """The standard bottleneck residual block without the last activation
    layer.

    It contains 3 conv layers with kernels 1x1, 3x3, 1x1.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        bottleneck_channels,
        norm='LN',
        act_layer=nn.GELU,
    ):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            bottleneck_channels (int): number of output channels for the 3x3
                "bottleneck" conv layers.
            norm (str or callable): normalization for all conv layers.
                See :func:`layers.get_norm` for supported format.
            act_layer (callable): activation for all conv layers.
        """
        super().__init__(in_channels, out_channels, 1)

        self.conv1 = Conv2d(in_channels, bottleneck_channels, 1, bias=False)
        self.norm1 = get_norm(norm, bottleneck_channels)
        self.act1 = act_layer()

        self.conv2 = Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            3,
            padding=1,
            bias=False,
        )
        self.norm2 = get_norm(norm, bottleneck_channels)
        self.act2 = act_layer()

        self.conv3 = Conv2d(bottleneck_channels, out_channels, 1, bias=False)
        self.norm3 = get_norm(norm, out_channels)

        for layer in [self.conv1, self.conv2, self.conv3]:
            weight_init.c2_msra_fill(layer)
        for layer in [self.norm1, self.norm2]:
            layer.weight.data.fill_(1.0)
            layer.bias.data.zero_()
        # zero init last norm layer.
        self.norm3.weight.data.zero_()
        self.norm3.bias.data.zero_()

    def forward(self, x):
        out = x
        for layer in self.children():
            out = layer(out)

        out = x + out
        return out


class Block(nn.Module):
    """Transformer blocks with support of window attention and residual
    propagation blocks."""

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        window_size=0,
        use_residual_block=False,
        input_size=None,
    ):
        """
        Args: dim (int): Number of input channels. num_heads (int): Number
        of attention heads in each ViT block. mlp_ratio (float): Ratio of
        mlp hidden dim to embedding dim. qkv_bias (bool): If True,
        add a learnable bias to query, key, value. drop_path (float):
        Stochastic depth rate. norm_layer (nn.Module): Normalization layer.
        act_layer (nn.Module): Activation layer. use_rel_pos (bool): If
        True, add relative positional embeddings to the attention map.
        rel_pos_zero_init (bool): If True, zero initialize relative
        positional parameters. window_size (int): Window size for window
        attention blocks. If it equals 0, then not use window attention.
        use_residual_block (bool): If True, use a residual block after the
        MLP block. input_size (int or None): Input resolution for
        calculating the relative positional parameter size.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else
            (window_size, window_size),
        )

        self.drop_path = DropPath(
            drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer)

        self.window_size = window_size

        self.use_residual_block = use_residual_block
        if use_residual_block:
            # Use a residual block with bottleneck channel as dim // 2
            self.residual = ResBottleneckBlock(
                in_channels=dim,
                out_channels=dim,
                bottleneck_channels=dim // 2,
                norm='LN',
                act_layer=act_layer,
            )

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        if self.use_residual_block:
            x = self.residual(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        return x


class ViT(Backbone):
    """This module implements Vision Transformer (ViT) backbone in
    :paper:`vitdet`.

    "Exploring Plain Vision Transformer Backbones for Object Detection",
    https://arxiv.org/abs/2203.16527
    """

    def __init__(
        self,
        img_size=1024,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        use_abs_pos=True,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        window_size=0,
        window_block_indexes=(),
        residual_block_indexes=(),
        use_act_checkpoint=True,
        pretrain_img_size=224,
        pretrain_use_cls_token=True,
        out_feature='last_feat',
    ):
        """
        Args: img_size (int): Input image size. patch_size (int): Patch
        size. in_chans (int): Number of input image channels. embed_dim (
        int): Patch embedding dimension. depth (int): Depth of ViT.
        num_heads (int): Number of attention heads in each ViT block.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool): If True, add a learnable bias to query, key,
        value. drop_path_rate (float): Stochastic depth rate. norm_layer (
        nn.Module): Normalization layer. act_layer (nn.Module): Activation
        layer. use_abs_pos (bool): If True, use absolute positional
        embeddings. use_rel_pos (bool): If True, add relative positional
        embeddings to the attention map. rel_pos_zero_init (bool): If True,
        zero initialize relative positional parameters. window_size (int):
        Window size for window attention blocks. window_block_indexes (
        list): Indexes for blocks using window attention.
        residual_block_indexes (list): Indexes for blocks using conv
        propagation. use_act_checkpoint (bool): If True, use activation
        checkpointing. pretrain_img_size (int): input image size for
        pretraining models. pretrain_use_cls_token (bool): If True,
        pretrainig models use class token. out_feature (str): name of the
        feature from the last block.
        """
        super().__init__()
        self.pretrain_use_cls_token = pretrain_use_cls_token
        self.use_act_checkpoint = use_act_checkpoint

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image
            # size.
            num_patches = (pretrain_img_size // patch_size) * (
                pretrain_img_size // patch_size)
            num_positions = (num_patches +
                             1) if pretrain_use_cls_token else num_patches
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_positions, embed_dim))
        else:
            self.pos_embed = None

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i in window_block_indexes else 0,
                use_residual_block=i in residual_block_indexes,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(block)

        self._out_feature_channels = {out_feature: embed_dim}
        self._out_feature_strides = {out_feature: patch_size}
        self._out_features = [out_feature]

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + get_abs_pos(self.pos_embed, self.pretrain_use_cls_token,
                                (x.shape[1], x.shape[2]))

        for blk in self.blocks:
            if self.use_act_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        return x.permute(0, 3, 1, 2)


class ViT_FPN(Backbone):

    def __init__(self,
                 bottom_up=None,
                 top_block=None,
                 out_channels=None,
                 strides=None,
                 vit_out_dim=None):
        super(ViT_FPN, self).__init__()
        assert isinstance(bottom_up, Backbone)
        self.bottom_up = bottom_up
        self.top_block = top_block

        self._out_feature_strides = {
            'p{}'.format(int(math.log2(s))): s
            for s in strides
        }
        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {
            k: out_channels
            for k in self._out_features
        }
        self._size_divisibility = strides[2]

        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.fpn_stride_16_8 = nn.ConvTranspose2d(
            vit_out_dim, vit_out_dim, 2, stride=2, bias=False)
        self.fpn_stride8_conv1 = nn.Conv2d(
            in_channels=vit_out_dim,
            out_channels=out_channels,
            kernel_size=1,
            bias=False)
        self.fpn_stride8_norm1 = nn.LayerNorm(out_channels)
        self.fpn_stride8_conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        self.fpn_stride8_norm2 = nn.LayerNorm(out_channels)

        self.fpn_stride16_conv1 = nn.Conv2d(
            in_channels=vit_out_dim,
            out_channels=out_channels,
            kernel_size=1,
            bias=False)
        self.fpn_stride16_norm1 = nn.LayerNorm(out_channels)
        self.fpn_stride16_conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        self.fpn_stride16_norm2 = nn.LayerNorm(out_channels)

        self.fpn_stride32_conv1 = nn.Conv2d(
            in_channels=vit_out_dim,
            out_channels=out_channels,
            kernel_size=1,
            bias=False)
        self.fpn_stride32_norm1 = nn.LayerNorm(out_channels)
        self.fpn_stride32_conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        self.fpn_stride32_norm2 = nn.LayerNorm(out_channels)

    def forward(self, x):
        vit_output_featuremap = self.bottom_up(x)

        stride8_feature = self.fpn_stride_16_8(vit_output_featuremap)
        stride8_feature = self.fpn_stride8_norm1(
            self.fpn_stride8_conv1(stride8_feature).permute(0, 2, 3,
                                                            1)).permute(
                                                                0, 3, 1, 2)
        stride8_feature = self.fpn_stride8_norm2(
            self.fpn_stride8_conv2(stride8_feature).permute(0, 2, 3,
                                                            1)).permute(
                                                                0, 3, 1, 2)

        stride32_feature = self.maxpool(vit_output_featuremap)
        stride32_feature = self.fpn_stride32_norm1(
            self.fpn_stride32_conv1(stride32_feature).permute(0, 2, 3,
                                                              1)).permute(
                                                                  0, 3, 1, 2)
        stride32_feature = self.fpn_stride32_norm2(
            self.fpn_stride32_conv2(stride32_feature).permute(0, 2, 3,
                                                              1)).permute(
                                                                  0, 3, 1, 2)

        stride16_feature = self.fpn_stride16_norm1(
            self.fpn_stride16_conv1(vit_output_featuremap).permute(
                0, 2, 3, 1)).permute(0, 3, 1, 2)
        stride16_feature = self.fpn_stride16_norm2(
            self.fpn_stride16_conv2(stride16_feature).permute(0, 2, 3,
                                                              1)).permute(
                                                                  0, 3, 1, 2)

        results = [stride8_feature, stride16_feature, stride32_feature]

        results.extend(self.top_block(stride32_feature))

        assert len(self._out_features) == len(results)
        fpn_out = {f: res for f, res in zip(self._out_features, results)}

        return fpn_out

    @property
    def size_divisibility(self):
        return self._size_divisibility

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name])
            for name in self._out_features
        }


@BACKBONE_REGISTRY.register()
def build_vit_fpn_backbone(cfg, input_shape: ShapeSpec):
    embed_dim = 768
    vit_out_dim = embed_dim
    bottom_up = ViT(  # Single-scale ViT backbone
        img_size=1024,
        patch_size=16,
        embed_dim=embed_dim,
        depth=12,
        num_heads=12,
        drop_path_rate=0.1,
        window_size=14,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        window_block_indexes=[
            # 2, 5, 8 11 for global attention
            0,
            1,
            3,
            4,
            6,
            7,
            9,
            10,
        ],
        residual_block_indexes=[],
        use_act_checkpoint=cfg.USE_ACT_CHECKPOINT,
        use_rel_pos=True,
        out_feature='last_feat',
    )

    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    assert out_channels == 256 or out_channels == 768 or out_channels == 1024
    backbone = ViT_FPN(
        bottom_up=bottom_up,
        top_block=LastLevelP6P7_P5(out_channels, out_channels),
        out_channels=out_channels,
        strides=[8, 16, 32, 64, 128],
        vit_out_dim=vit_out_dim)
    return backbone


@BACKBONE_REGISTRY.register()
def build_vit_fpn_backbone_large(cfg, input_shape: ShapeSpec):
    window_block_indexes = (
        list(range(0, 5)) + list(range(6, 11)) + list(range(12, 17)) +
        list(range(18, 23)))
    embed_dim = 1024
    vit_out_dim = embed_dim
    bottom_up = ViT(  # Single-scale ViT backbone
        img_size=1024,
        patch_size=16,
        embed_dim=embed_dim,
        depth=24,
        num_heads=16,
        drop_path_rate=0.4,
        window_size=14,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        window_block_indexes=window_block_indexes,
        residual_block_indexes=[],
        use_act_checkpoint=cfg.USE_ACT_CHECKPOINT,
        use_rel_pos=True,
        out_feature='last_feat',
    )

    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    assert out_channels == 256 or out_channels == 768 or out_channels == 1024
    backbone = ViT_FPN(
        bottom_up=bottom_up,
        top_block=LastLevelP6P7_P5(out_channels, out_channels),
        out_channels=out_channels,
        strides=[8, 16, 32, 64, 128],
        vit_out_dim=vit_out_dim)
    return backbone


@BACKBONE_REGISTRY.register()
def build_vit_fpn_backbone_huge(cfg, input_shape: ShapeSpec):
    window_block_indexes = (
        list(range(0, 7)) + list(range(8, 15)) + list(range(16, 23)) +
        list(range(24, 31)))
    embed_dim = 1280
    vit_out_dim = embed_dim
    bottom_up = ViT(  # Single-scale ViT backbone
        img_size=1024,
        patch_size=16,
        embed_dim=embed_dim,
        depth=32,
        num_heads=16,
        drop_path_rate=0.5,
        window_size=14,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        window_block_indexes=window_block_indexes,
        residual_block_indexes=[],
        use_act_checkpoint=cfg.USE_ACT_CHECKPOINT,
        use_rel_pos=True,
        out_feature='last_feat',
    )

    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    assert out_channels == 256 or out_channels == 768 or out_channels == 1024
    backbone = ViT_FPN(
        bottom_up=bottom_up,
        top_block=LastLevelP6P7_P5(out_channels, out_channels),
        out_channels=out_channels,
        strides=[8, 16, 32, 64, 128],
        vit_out_dim=vit_out_dim)
    return backbone
