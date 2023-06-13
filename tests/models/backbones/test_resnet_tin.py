# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
import torch.nn as nn

from mmaction.models import ResNetTIN
from mmaction.testing import generate_backbone_demo_inputs


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires CUDA support')
def test_resnet_tin_backbone():
    """Test resnet_tin backbone."""
    with pytest.raises(AssertionError):
        # num_segments should be positive
        resnet_tin = ResNetTIN(50, num_segments=-1)
        resnet_tin.init_weights()

    from mmaction.models.backbones.resnet_tin import (CombineNet,
                                                      TemporalInterlace)

    # resnet_tin with normal config
    resnet_tin = ResNetTIN(50)
    resnet_tin.init_weights()
    for layer_name in resnet_tin.res_layers:
        layer = getattr(resnet_tin, layer_name)
        blocks = list(layer.children())
        for block in blocks:
            assert isinstance(block.conv1.conv, CombineNet)
            assert isinstance(block.conv1.conv.net1, TemporalInterlace)
            assert (
                block.conv1.conv.net1.num_segments == resnet_tin.num_segments)
            assert block.conv1.conv.net1.shift_div == resnet_tin.shift_div

    # resnet_tin with partial batchnorm
    resnet_tin_pbn = ResNetTIN(50, partial_bn=True)
    resnet_tin_pbn.train()
    count_bn = 0
    for m in resnet_tin_pbn.modules():
        if isinstance(m, nn.BatchNorm2d):
            count_bn += 1
            if count_bn >= 2:
                assert m.training is False
                assert m.weight.requires_grad is False
                assert m.bias.requires_grad is False
            else:
                assert m.training is True
                assert m.weight.requires_grad is True
                assert m.bias.requires_grad is True

    input_shape = (8, 3, 64, 64)
    imgs = generate_backbone_demo_inputs(input_shape).cuda()
    resnet_tin = resnet_tin.cuda()

    # resnet_tin with normal cfg inference
    feat = resnet_tin(imgs)
    assert feat.shape == torch.Size([8, 2048, 2, 2])
