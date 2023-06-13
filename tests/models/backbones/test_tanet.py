# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmaction.models import TANet
from mmaction.testing import generate_backbone_demo_inputs


def test_tanet_backbone():
    """Test tanet backbone."""
    with pytest.raises(NotImplementedError):
        # TA-Blocks are only based on Bottleneck block now
        tanet_18 = TANet(18, 8)
        tanet_18.init_weights()

    from mmaction.models.backbones.resnet import Bottleneck
    from mmaction.models.backbones.tanet import TABlock

    # tanet with depth 50
    tanet_50 = TANet(50, 8)
    tanet_50.init_weights()

    for layer_name in tanet_50.res_layers:
        layer = getattr(tanet_50, layer_name)
        blocks = list(layer.children())
        for block in blocks:
            assert isinstance(block, TABlock)
            assert isinstance(block.block, Bottleneck)
            assert block.tam.num_segments == block.num_segments
            assert block.tam.in_channels == block.block.conv1.out_channels

    input_shape = (8, 3, 64, 64)
    imgs = generate_backbone_demo_inputs(input_shape)
    feat = tanet_50(imgs)
    assert feat.shape == torch.Size([8, 2048, 2, 2])

    input_shape = (16, 3, 32, 32)
    imgs = generate_backbone_demo_inputs(input_shape)
    feat = tanet_50(imgs)
    assert feat.shape == torch.Size([16, 2048, 1, 1])
