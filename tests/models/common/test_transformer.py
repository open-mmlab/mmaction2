# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmengine.testing import assert_params_all_zeros

from mmaction.models.common import (DividedSpatialAttentionWithNorm,
                                    DividedTemporalAttentionWithNorm,
                                    FFNWithNorm)


def test_divided_temporal_attention_with_norm():
    _cfg = dict(embed_dims=768, num_heads=12, num_frames=8)
    divided_temporal_attention = DividedTemporalAttentionWithNorm(**_cfg)
    assert isinstance(divided_temporal_attention.norm, nn.LayerNorm)
    assert assert_params_all_zeros(divided_temporal_attention.temporal_fc)

    x = torch.rand(1, 1 + 8 * 14 * 14, 768)
    output = divided_temporal_attention(x)
    assert output.shape == torch.Size([1, 1 + 8 * 14 * 14, 768])


def test_divided_spatial_attention_with_norm():
    _cfg = dict(embed_dims=512, num_heads=8, num_frames=4, dropout_layer=None)
    divided_spatial_attention = DividedSpatialAttentionWithNorm(**_cfg)
    assert isinstance(divided_spatial_attention.dropout_layer, nn.Identity)
    assert isinstance(divided_spatial_attention.norm, nn.LayerNorm)

    x = torch.rand(1, 1 + 4 * 14 * 14, 512)
    output = divided_spatial_attention(x)
    assert output.shape == torch.Size([1, 1 + 4 * 14 * 14, 512])


def test_ffn_with_norm():
    _cfg = dict(
        embed_dims=256, feedforward_channels=256 * 2, norm_cfg=dict(type='LN'))
    ffn_with_norm = FFNWithNorm(**_cfg)
    assert isinstance(ffn_with_norm.norm, nn.LayerNorm)

    x = torch.rand(1, 1 + 4 * 14 * 14, 256)
    output = ffn_with_norm(x)
    assert output.shape == torch.Size([1, 1 + 4 * 14 * 14, 256])
