# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks import DropPath
from mmcv.cnn.bricks.transformer import FFN, PatchEmbed
from mmengine.model import BaseModule, ModuleList
from torch import Tensor, nn

from mmaction.registry import MODELS
from mmaction.utils import ConfigType, OptConfigType


class Attention(BaseModule):
    """Multi-head Self-attention.

    Args:
        embed_dims (int): Dimensions of embedding.
        num_heads (int): Number of parallel attention heads.
        qkv_bias (bool): If True, add a learnable bias to q and v.
            Defaults to True.
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        attn_drop_rate (float): Dropout ratio of attention weight.
            Defaults to 0.
        drop_rate (float): Dropout ratio of output. Defaults to 0.
        init_cfg (dict or ConfigDict, optional): The Config
            for initialization. Defaults to None.
    """

    def __init__(self,
                 embed_dims: int,
                 num_heads: int = 8,
                 qkv_bias: bool = True,
                 qk_scale: Optional[float] = None,
                 attn_drop_rate: float = 0.,
                 drop_rate: float = 0.,
                 init_cfg: OptConfigType = None,
                 **kwargs) -> None:
        super().__init__(init_cfg=init_cfg)
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads

        self.scale = qk_scale or head_embed_dims**-0.5

        if qkv_bias:
            self._init_qv_bias()

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=False)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(drop_rate)

    def _init_qv_bias(self) -> None:
        self.q_bias = nn.Parameter(torch.zeros(self.embed_dims))
        self.v_bias = nn.Parameter(torch.zeros(self.embed_dims))

    def forward(self, x: Tensor) -> Tensor:
        """Defines the computation performed at every call.

        Args:
            x (Tensor): The input data with size of (B, N, C).
        Returns:
            Tensor: The output of the attention block, same size as inputs.
        """
        B, N, C = x.shape

        if hasattr(self, 'q_bias'):
            k_bias = torch.zeros_like(self.v_bias, requires_grad=False)
            qkv_bias = torch.cat((self.q_bias, k_bias, self.v_bias))
            qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        else:
            qkv = self.qkv(x)

        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(BaseModule):
    """The basic block in the Vision Transformer.

    Args:
        embed_dims (int): Dimensions of embedding.
        num_heads (int): Number of parallel attention heads.
        mlp_ratio (int): The ratio between the hidden layer and the
            input layer in the FFN. Defaults to 4.
        qkv_bias (bool): If True, add a learnable bias to q and v.
            Defaults to True.
        qk_scale (float): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        drop_rate (float): Dropout ratio of output. Defaults to 0.
        attn_drop_rate (float): Dropout ratio of attention weight.
            Defaults to 0.
        drop_path_rate (float): Dropout ratio of the residual branch.
            Defaults to 0.
        init_values (float): Value to init the multiplier of the
            residual branch. Defaults to 0.
        act_cfg (dict or ConfigDict): Config for activation layer in FFN.
            Defaults to `dict(type='GELU')`.
        norm_cfg (dict or ConfigDict): Config for norm layers.
            Defaults to `dict(type='LN', eps=1e-6)`.
        init_cfg (dict or ConfigDict, optional): The Config
            for initialization. Defaults to None.
    """

    def __init__(self,
                 embed_dims: int,
                 num_heads: int,
                 mlp_ratio: int = 4.,
                 qkv_bias: bool = True,
                 qk_scale: Optional[float] = None,
                 drop_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 init_values: float = 0.0,
                 act_cfg: ConfigType = dict(type='GELU'),
                 norm_cfg: ConfigType = dict(type='LN', eps=1e-6),
                 init_cfg: OptConfigType = None,
                 **kwargs) -> None:
        super().__init__(init_cfg=init_cfg)
        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.attn = Attention(
            embed_dims,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            drop_rate=drop_rate)

        self.drop_path = nn.Identity()
        if drop_path_rate > 0.:
            self.drop_path = DropPath(drop_path_rate)
        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]

        mlp_hidden_dim = int(embed_dims * mlp_ratio)
        self.mlp = FFN(
            embed_dims=embed_dims,
            feedforward_channels=mlp_hidden_dim,
            act_cfg=act_cfg,
            ffn_drop=drop_rate,
            add_identity=False)

        self._init_gammas(init_values, embed_dims)

    def _init_gammas(self, init_values: float, dim: int) -> None:
        if type(init_values) == float and init_values > 0:
            self.gamma_1 = nn.Parameter(
                init_values * torch.ones(dim), requires_grad=True)
            self.gamma_2 = nn.Parameter(
                init_values * torch.ones(dim), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        """Defines the computation performed at every call.

        Args:
            x (Tensor): The input data with size of (B, N, C).
        Returns:
            Tensor: The output of the transformer block, same size as inputs.
        """
        if hasattr(self, 'gamma_1'):
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def get_sinusoid_encoding(n_position: int, embed_dims: int) -> Tensor:
    """Generate sinusoid encoding table.

    Sinusoid encoding is a kind of relative position encoding method came from
    `Attention Is All You Need<https://arxiv.org/abs/1706.03762>`_.
    Args:
        n_position (int): The length of the input token.
        embed_dims (int): The position embedding dimension.
    Returns:
        :obj:`torch.FloatTensor`: The sinusoid encoding table of size
        (1, n_position, embed_dims)
    """

    vec = torch.arange(embed_dims, dtype=torch.float64)
    vec = (vec - vec % 2) / embed_dims
    vec = torch.pow(10000, -vec).view(1, -1)

    sinusoid_table = torch.arange(n_position).view(-1, 1) * vec
    sinusoid_table[:, 0::2].sin_()  # dim 2i
    sinusoid_table[:, 1::2].cos_()  # dim 2i+1

    sinusoid_table = sinusoid_table.to(torch.float32)

    return sinusoid_table.unsqueeze(0)


@MODELS.register_module()
class VisionTransformer(BaseModule):
    """Vision Transformer with support for patch or hybrid CNN input stage. An
    impl of `VideoMAE: Masked Autoencoders are Data-Efficient Learners for
    Self-Supervised Video Pre-Training <https://arxiv.org/pdf/2203.12602.pdf>`_

    Args:
        img_size (int or tuple): Size of input image.
            Defaults to 224.
        patch_size (int): Spatial size of one patch. Defaults to 16.
        in_channels (int): The number of channels of he input.
            Defaults to 3.
        embed_dims (int): Dimensions of embedding. Defaults to 768.
        depth (int): number of blocks in the transformer.
            Defaults to 12.
        num_heads (int): Number of parallel attention heads in
            TransformerCoder. Defaults to 12.
        mlp_ratio (int): The ratio between the hidden layer and the
            input layer in the FFN. Defaults to 4.
        qkv_bias (bool): If True, add a learnable bias to q and v.
            Defaults to True.
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        drop_rate (float): Dropout ratio of output. Defaults to 0.
        attn_drop_rate (float): Dropout ratio of attention weight.
            Defaults to 0.
        drop_path_rate (float): Dropout ratio of the residual branch.
            Defaults to 0.
        norm_cfg (dict or Configdict): Config for norm layers.
            Defaults to `dict(type='LN', eps=1e-6)`.
        init_values (float): Value to init the multiplier of the residual
            branch. Defaults to 0.
        use_learnable_pos_emb (bool): If True, use learnable positional
            embedding, othersize use sinusoid encoding. Defaults to False.
        num_frames (int): Number of frames in the video. Defaults to 16.
        tubelet_size (int): Temporal size of one patch. Defaults to 2.
        use_mean_pooling (bool): If True, take the mean pooling over all
            positions. Defaults to True.
        pretrained (str, optional): Name of pretrained model. Default: None.
        return_feat_map (bool): If True, return the feature in the shape of
            `[B, C, T, H, W]`. Defaults to False.
        init_cfg (dict or list[dict]): Initialization config dict. Defaults to
            ``[
            dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
            dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
            ]``.
    """

    def __init__(self,
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 embed_dims: int = 768,
                 depth: int = 12,
                 num_heads: int = 12,
                 mlp_ratio: int = 4.,
                 qkv_bias: bool = True,
                 qk_scale: int = None,
                 drop_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 norm_cfg: ConfigType = dict(type='LN', eps=1e-6),
                 init_values: int = 0.,
                 use_learnable_pos_emb: bool = False,
                 num_frames: int = 16,
                 tubelet_size: int = 2,
                 use_mean_pooling: int = True,
                 pretrained: Optional[str] = None,
                 return_feat_map: bool = False,
                 init_cfg: Optional[Union[Dict, List[Dict]]] = [
                     dict(
                         type='TruncNormal', layer='Linear', std=0.02,
                         bias=0.),
                     dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
                 ],
                 **kwargs) -> None:

        if pretrained:
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        super().__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims
        self.patch_size = patch_size

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims,
            conv_type='Conv3d',
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
            padding=(0, 0, 0),
            dilation=(1, 1, 1))

        grid_size = img_size // patch_size
        num_patches = grid_size**2 * (num_frames // tubelet_size)
        self.grid_size = (grid_size, grid_size)

        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dims))
            nn.init.trunc_normal_(self.pos_embed, std=.02)
        else:
            # sine-cosine positional embeddings is on the way
            pos_embed = get_sinusoid_encoding(num_patches, embed_dims)
            self.register_buffer('pos_embed', pos_embed)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = ModuleList([
            Block(
                embed_dims=embed_dims,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[i],
                norm_cfg=norm_cfg,
                init_values=init_values) for i in range(depth)
        ])

        if use_mean_pooling:
            self.norm = nn.Identity()
            self.fc_norm = build_norm_layer(norm_cfg, embed_dims)[1]
        else:
            self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
            self.fc_norm = None

        self.return_feat_map = return_feat_map

    def forward(self, x: Tensor) -> Tensor:
        """Defines the computation performed at every call.

        Args:
            x (Tensor): The input data.
        Returns:
            Tensor: The feature of the input
                samples extracted by the backbone.
        """
        b, _, _, h, w = x.shape
        h //= self.patch_size
        w //= self.patch_size
        x = self.patch_embed(x)[0]
        if (h, w) != self.grid_size:
            pos_embed = self.pos_embed.reshape(-1, *self.grid_size,
                                               self.embed_dims)
            pos_embed = pos_embed.permute(0, 3, 1, 2)
            pos_embed = F.interpolate(
                pos_embed, size=(h, w), mode='bicubic', align_corners=False)
            pos_embed = pos_embed.permute(0, 2, 3, 1).flatten(1, 2)
            pos_embed = pos_embed.reshape(1, -1, self.embed_dims)
        else:
            pos_embed = self.pos_embed

        x = x + pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        if self.return_feat_map:
            x = x.reshape(b, -1, h, w, self.embed_dims)
            x = x.permute(0, 4, 1, 2, 3)
            return x

        if self.fc_norm is not None:
            return self.fc_norm(x.mean(1))

        return x[:, 0]
