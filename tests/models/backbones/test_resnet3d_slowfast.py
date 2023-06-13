# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm

from mmaction.models import ResNet3dSlowFast
from mmaction.testing import generate_backbone_demo_inputs


def test_slowfast_backbone():
    """Test SlowFast backbone."""
    with pytest.raises(TypeError):
        # cfg should be a dict
        ResNet3dSlowFast(slow_pathway=list(['foo', 'bar']))
    with pytest.raises(KeyError):
        # pathway type should be implemented
        ResNet3dSlowFast(slow_pathway=dict(type='resnext'))

    # test slowfast with slow inflated
    sf_50_inflate = ResNet3dSlowFast(
        slow_pathway=dict(
            type='resnet3d',
            depth=50,
            pretrained='torchvision://resnet50',
            pretrained2d=True,
            lateral=True,
            conv1_kernel=(1, 7, 7),
            dilations=(1, 1, 1, 1),
            conv1_stride_t=1,
            pool1_stride_t=1,
            inflate=(0, 0, 1, 1)))
    sf_50_inflate.init_weights()
    sf_50_inflate.train()

    # test slowfast with no lateral connection
    sf_50_wo_lateral = ResNet3dSlowFast(
        None,
        slow_pathway=dict(
            type='resnet3d',
            depth=50,
            pretrained=None,
            lateral=False,
            conv1_kernel=(1, 7, 7),
            dilations=(1, 1, 1, 1),
            conv1_stride_t=1,
            pool1_stride_t=1,
            inflate=(0, 0, 1, 1)))
    sf_50_wo_lateral.init_weights()
    sf_50_wo_lateral.train()

    # slowfast w/o lateral connection inference test
    input_shape = (1, 3, 8, 64, 64)
    imgs = generate_backbone_demo_inputs(input_shape)
    feat = sf_50_wo_lateral(imgs)

    assert isinstance(feat, tuple)
    assert feat[0].shape == torch.Size([1, 2048, 1, 2, 2])
    assert feat[1].shape == torch.Size([1, 256, 8, 2, 2])

    # test slowfast with frozen stages config
    frozen_slow = 3
    sf_50 = ResNet3dSlowFast(
        None,
        slow_pathway=dict(
            type='resnet3d',
            depth=50,
            pretrained=None,
            pretrained2d=True,
            lateral=True,
            conv1_kernel=(1, 7, 7),
            dilations=(1, 1, 1, 1),
            conv1_stride_t=1,
            pool1_stride_t=1,
            inflate=(0, 0, 1, 1),
            frozen_stages=frozen_slow))
    sf_50.init_weights()
    sf_50.train()

    for stage in range(1, sf_50.slow_path.num_stages):
        lateral_name = sf_50.slow_path.lateral_connections[stage - 1]
        conv_lateral = getattr(sf_50.slow_path, lateral_name)
        for mod in conv_lateral.modules():
            if isinstance(mod, _BatchNorm):
                if stage <= frozen_slow:
                    assert mod.training is False
                else:
                    assert mod.training is True
        for param in conv_lateral.parameters():
            if stage <= frozen_slow:
                assert param.requires_grad is False
            else:
                assert param.requires_grad is True

    # test slowfast with normal config
    sf_50 = ResNet3dSlowFast()
    sf_50.init_weights()
    sf_50.train()

    # slowfast inference test
    input_shape = (1, 3, 8, 64, 64)
    imgs = generate_backbone_demo_inputs(input_shape)
    feat = sf_50(imgs)

    assert isinstance(feat, tuple)
    assert feat[0].shape == torch.Size([1, 2048, 1, 2, 2])
    assert feat[1].shape == torch.Size([1, 256, 8, 2, 2])
