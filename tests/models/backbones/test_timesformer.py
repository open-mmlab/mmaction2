# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmaction.models import TimeSformer
from mmaction.testing import generate_backbone_demo_inputs


def test_timesformer_backbone():
    input_shape = (1, 3, 8, 64, 64)
    imgs = generate_backbone_demo_inputs(input_shape)

    # divided_space_time
    timesformer = TimeSformer(
        8, 64, 16, embed_dims=768, attention_type='divided_space_time')
    timesformer.init_weights()
    from mmaction.models.common import (DividedSpatialAttentionWithNorm,
                                        DividedTemporalAttentionWithNorm,
                                        FFNWithNorm)
    assert isinstance(timesformer.transformer_layers.layers[0].attentions[0],
                      DividedTemporalAttentionWithNorm)
    assert isinstance(timesformer.transformer_layers.layers[11].attentions[1],
                      DividedSpatialAttentionWithNorm)
    assert isinstance(timesformer.transformer_layers.layers[0].ffns[0],
                      FFNWithNorm)
    assert hasattr(timesformer, 'time_embed')
    assert timesformer.patch_embed.num_patches == 16

    cls_tokens = timesformer(imgs)
    assert cls_tokens.shape == torch.Size([1, 768])

    # space_only
    timesformer = TimeSformer(
        8, 64, 16, embed_dims=512, num_heads=8, attention_type='space_only')
    timesformer.init_weights()

    assert not hasattr(timesformer, 'time_embed')
    assert timesformer.patch_embed.num_patches == 16

    cls_tokens = timesformer(imgs)
    assert cls_tokens.shape == torch.Size([1, 512])

    # joint_space_time
    input_shape = (1, 3, 2, 64, 64)
    imgs = generate_backbone_demo_inputs(input_shape)
    timesformer = TimeSformer(
        2,
        64,
        8,
        embed_dims=256,
        num_heads=8,
        attention_type='joint_space_time')
    timesformer.init_weights()

    assert hasattr(timesformer, 'time_embed')
    assert timesformer.patch_embed.num_patches == 64

    cls_tokens = timesformer(imgs)
    assert cls_tokens.shape == torch.Size([1, 256])

    with pytest.raises(AssertionError):
        # unsupported attention type
        timesformer = TimeSformer(
            8, 64, 16, attention_type='wrong_attention_type')

    with pytest.raises(AssertionError):
        # Wrong transformer_layers type
        timesformer = TimeSformer(8, 64, 16, transformer_layers='wrong_type')
