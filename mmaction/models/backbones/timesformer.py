# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from mmcv import ConfigDict
from mmcv.cnn import build_conv_layer, build_norm_layer, kaiming_init
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.cnn.utils.weight_init import trunc_normal_
from mmcv.runner import _load_checkpoint, load_state_dict
from torch.nn.modules.utils import _pair

from ...utils import get_root_logger
from ..builder import BACKBONES


class PatchEmbed(nn.Module):
    """Image to Patch Embedding.

    Args:
        img_size (int | tuple): Size of input image.
        patch_size (int): Size of one patch.
        in_channels (int): Channel num of input features. Defaults to 3.
        embed_dims (int): Dimensions of embedding. Defaults to 768.
        conv_cfg (dict | None): Config dict for convolution layer. Defaults to
            `dict(type='Conv2d')`.
    """

    def __init__(self,
                 img_size,
                 patch_size,
                 in_channels=3,
                 embed_dims=768,
                 conv_cfg=dict(type='Conv2d')):
        super().__init__()
        self.img_size = _pair(img_size)
        self.patch_size = _pair(patch_size)

        num_patches = (self.img_size[1] // self.patch_size[1]) * (
            self.img_size[0] // self.patch_size[0])
        assert num_patches * self.patch_size[0] * self.patch_size[1] == \
               self.img_size[0] * self.img_size[1], \
               'The image size H*W must be divisible by patch size'
        self.num_patches = num_patches

        # Use conv layer to embed
        self.projection = build_conv_layer(
            conv_cfg,
            in_channels,
            embed_dims,
            kernel_size=patch_size,
            stride=patch_size)

        self.init_weights()

    def init_weights(self):
        # Lecun norm from ClassyVision
        kaiming_init(self.projection, mode='fan_in', nonlinearity='linear')

    def forward(self, x):
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.projection(x).flatten(2).transpose(1, 2)
        return x


@BACKBONES.register_module()
class TimeSformer(nn.Module):
    """TimeSformer. A PyTorch impl of `Is Space-Time Attention All You Need for
    Video Understanding? <https://arxiv.org/abs/2102.05095>`_

    Args:
        num_frames (int): Number of frames in the video.
        img_size (int | tuple): Size of input image.
        patch_size (int): Size of one patch.
        pretrained (str | None): Name of pretrained model. Default: None.
        embed_dims (int): Dimensions of embedding. Defaults to 768.
        num_heads (int): Number of parallel attention heads in
            TransformerCoder. Defaults to 12.
        num_transformer_layers (int): Number of transformer layers. Defaults to
            12.
        in_channels (int): Channel num of input features. Defaults to 3.
        dropout_ratio (float): Probability of dropout layer. Defaults to 0..
        transformer_layers (list[obj:`mmcv.ConfigDict`] |
            obj:`mmcv.ConfigDict` | None): Config of transformerlayer in
            TransformerCoder. If it is obj:`mmcv.ConfigDict`, it would be
            repeated `num_transformer_layers` times to a
            list[obj:`mmcv.ConfigDict`]. Defaults to None.
        attention_type (str): Type of attentions in TransformerCoder. Choices
            are 'divided_space_time', 'space_only' and 'joint_space_time'.
            Defaults to 'divided_space_time'.
        norm_cfg (dict): Config for norm layers. Defaults to
            `dict(type='LN', eps=1e-6)`.
    """
    supported_attention_types = [
        'divided_space_time', 'space_only', 'joint_space_time'
    ]

    def __init__(self,
                 num_frames,
                 img_size,
                 patch_size,
                 pretrained=None,
                 embed_dims=768,
                 num_heads=12,
                 num_transformer_layers=12,
                 in_channels=3,
                 dropout_ratio=0.,
                 transformer_layers=None,
                 attention_type='divided_space_time',
                 norm_cfg=dict(type='LN', eps=1e-6),
                 **kwargs):
        super().__init__(**kwargs)
        assert attention_type in self.supported_attention_types, (
            f'Unsupported Attention Type {attention_type}!')
        assert transformer_layers is None or isinstance(
            transformer_layers, (dict, list))

        self.num_frames = num_frames
        self.pretrained = pretrained
        self.embed_dims = embed_dims
        self.num_transformer_layers = num_transformer_layers
        self.attention_type = attention_type

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dims=embed_dims)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dims))
        self.drop_after_pos = nn.Dropout(p=dropout_ratio)
        if self.attention_type != 'space_only':
            self.time_embed = nn.Parameter(
                torch.zeros(1, num_frames, embed_dims))
            self.drop_after_time = nn.Dropout(p=dropout_ratio)

        self.norm = build_norm_layer(norm_cfg, embed_dims)[1]

        if transformer_layers is None:
            # stochastic depth decay rule
            dpr = np.linspace(0, 0.1, num_transformer_layers)

            if self.attention_type == 'divided_space_time':
                _transformerlayers_cfg = [
                    dict(
                        type='BaseTransformerLayer',
                        attn_cfgs=[
                            dict(
                                type='DividedTemporalAttentionWithNorm',
                                embed_dims=embed_dims,
                                num_heads=num_heads,
                                num_frames=num_frames,
                                dropout_layer=dict(
                                    type='DropPath', drop_prob=dpr[i]),
                                norm_cfg=dict(type='LN', eps=1e-6)),
                            dict(
                                type='DividedSpatialAttentionWithNorm',
                                embed_dims=embed_dims,
                                num_heads=num_heads,
                                num_frames=num_frames,
                                dropout_layer=dict(
                                    type='DropPath', drop_prob=dpr[i]),
                                norm_cfg=dict(type='LN', eps=1e-6))
                        ],
                        ffn_cfgs=dict(
                            type='FFNWithNorm',
                            embed_dims=embed_dims,
                            feedforward_channels=embed_dims * 4,
                            num_fcs=2,
                            act_cfg=dict(type='GELU'),
                            dropout_layer=dict(
                                type='DropPath', drop_prob=dpr[i]),
                            norm_cfg=dict(type='LN', eps=1e-6)),
                        operation_order=('self_attn', 'self_attn', 'ffn'))
                    for i in range(num_transformer_layers)
                ]
            else:
                # Sapce Only & Joint Space Time
                _transformerlayers_cfg = [
                    dict(
                        type='BaseTransformerLayer',
                        attn_cfgs=[
                            dict(
                                type='MultiheadAttention',
                                embed_dims=embed_dims,
                                num_heads=num_heads,
                                batch_first=True,
                                dropout_layer=dict(
                                    type='DropPath', drop_prob=dpr[i]))
                        ],
                        ffn_cfgs=dict(
                            type='FFN',
                            embed_dims=embed_dims,
                            feedforward_channels=embed_dims * 4,
                            num_fcs=2,
                            act_cfg=dict(type='GELU'),
                            dropout_layer=dict(
                                type='DropPath', drop_prob=dpr[i])),
                        operation_order=('norm', 'self_attn', 'norm', 'ffn'),
                        norm_cfg=dict(type='LN', eps=1e-6),
                        batch_first=True)
                    for i in range(num_transformer_layers)
                ]

            transformer_layers = ConfigDict(
                dict(
                    type='TransformerLayerSequence',
                    transformerlayers=_transformerlayers_cfg,
                    num_layers=num_transformer_layers))

        self.transformer_layers = build_transformer_layer_sequence(
            transformer_layers)

    def init_weights(self, pretrained=None):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)

        if pretrained:
            self.pretrained = pretrained
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            logger.info(f'load model from: {self.pretrained}')

            state_dict = _load_checkpoint(self.pretrained)
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']

            if self.attention_type == 'divided_space_time':
                # modify the key names of norm layers
                old_state_dict_keys = list(state_dict.keys())
                for old_key in old_state_dict_keys:
                    if 'norms' in old_key:
                        new_key = old_key.replace('norms.0',
                                                  'attentions.0.norm')
                        new_key = new_key.replace('norms.1', 'ffns.0.norm')
                        state_dict[new_key] = state_dict.pop(old_key)

                # copy the parameters of space attention to time attention
                old_state_dict_keys = list(state_dict.keys())
                for old_key in old_state_dict_keys:
                    if 'attentions.0' in old_key:
                        new_key = old_key.replace('attentions.0',
                                                  'attentions.1')
                        state_dict[new_key] = state_dict[old_key].clone()

            load_state_dict(self, state_dict, strict=False, logger=logger)

    def forward(self, x):
        """Defines the computation performed at every call."""
        # x [batch_size * num_frames, num_patches, embed_dims]
        batches = x.shape[0]
        x = self.patch_embed(x)

        # x [batch_size * num_frames, num_patches + 1, embed_dims]
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.drop_after_pos(x)

        # Add Time Embedding
        if self.attention_type != 'space_only':
            # x [batch_size, num_patches * num_frames + 1, embed_dims]
            cls_tokens = x[:batches, 0, :].unsqueeze(1)
            x = rearrange(x[:, 1:, :], '(b t) p m -> (b p) t m', b=batches)
            x = x + self.time_embed
            x = self.drop_after_time(x)
            x = rearrange(x, '(b p) t m -> b (p t) m', b=batches)
            x = torch.cat((cls_tokens, x), dim=1)

        x = self.transformer_layers(x, None, None)

        if self.attention_type == 'space_only':
            # x [batch_size, num_patches + 1, embed_dims]
            x = x.view(-1, self.num_frames, *x.size()[-2:])
            x = torch.mean(x, 1)

        x = self.norm(x)

        # Return Class Token
        return x[:, 0]
