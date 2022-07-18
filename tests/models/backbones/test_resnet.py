# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
import torch.nn as nn
from mmcv.utils import _BatchNorm

from mmaction.models import ResNet
from ..base import check_norm_state, generate_backbone_demo_inputs


def test_resnet_backbone():
    """Test resnet backbone."""
    with pytest.raises(KeyError):
        # ResNet depth should be in [18, 34, 50, 101, 152]
        ResNet(20)

    with pytest.raises(AssertionError):
        # In ResNet: 1 <= num_stages <= 4
        ResNet(50, num_stages=0)

    with pytest.raises(AssertionError):
        # In ResNet: 1 <= num_stages <= 4
        ResNet(50, num_stages=5)

    with pytest.raises(AssertionError):
        # len(strides) == len(dilations) == num_stages
        ResNet(50, strides=(1, ), dilations=(1, 1), num_stages=3)

    with pytest.raises(TypeError):
        # pretrain must be a str
        resnet50 = ResNet(50, pretrained=0)
        resnet50.init_weights()

    with pytest.raises(AssertionError):
        # style must be in ['pytorch', 'caffe']
        ResNet(18, style='tensorflow')

    with pytest.raises(AssertionError):
        # assert not with_cp
        ResNet(18, with_cp=True)

    # resnet with depth 18, norm_eval False, initial weights
    resnet18 = ResNet(18)
    resnet18.init_weights()

    # resnet with depth 50, norm_eval True
    resnet50 = ResNet(50, norm_eval=True)
    resnet50.init_weights()
    resnet50.train()
    assert check_norm_state(resnet50.modules(), False)

    # resnet with depth 50, norm_eval True, pretrained
    resnet50_pretrain = ResNet(
        pretrained='torchvision://resnet50', depth=50, norm_eval=True)
    resnet50_pretrain.init_weights()
    resnet50_pretrain.train()
    assert check_norm_state(resnet50_pretrain.modules(), False)

    # resnet with depth 50, norm_eval True, frozen_stages 1
    frozen_stages = 1
    resnet50_frozen = ResNet(50, frozen_stages=frozen_stages)
    resnet50_frozen.init_weights()
    resnet50_frozen.train()
    assert resnet50_frozen.conv1.bn.training is False
    for layer in resnet50_frozen.conv1.modules():
        for param in layer.parameters():
            assert param.requires_grad is False
    for i in range(1, frozen_stages + 1):
        layer = getattr(resnet50_frozen, f'layer{i}')
        for mod in layer.modules():
            if isinstance(mod, _BatchNorm):
                assert mod.training is False
        for param in layer.parameters():
            assert param.requires_grad is False

    # resnet with depth 50, partial batchnorm
    resnet_pbn = ResNet(50, partial_bn=True)
    resnet_pbn.train()
    count_bn = 0
    for m in resnet_pbn.modules():
        if isinstance(m, nn.BatchNorm2d):
            count_bn += 1
            if count_bn >= 2:
                assert m.weight.requires_grad is False
                assert m.bias.requires_grad is False
                assert m.training is False
            else:
                assert m.weight.requires_grad is True
                assert m.bias.requires_grad is True
                assert m.training is True

    input_shape = (1, 3, 64, 64)
    imgs = generate_backbone_demo_inputs(input_shape)

    # resnet with depth 18 inference
    resnet18 = ResNet(18, norm_eval=False)
    resnet18.init_weights()
    resnet18.train()
    feat = resnet18(imgs)
    assert feat.shape == torch.Size([1, 512, 2, 2])

    # resnet with depth 50 inference
    resnet50 = ResNet(50, norm_eval=False)
    resnet50.init_weights()
    resnet50.train()
    feat = resnet50(imgs)
    assert feat.shape == torch.Size([1, 2048, 2, 2])

    # resnet with depth 50 in caffe style inference
    resnet50_caffe = ResNet(50, style='caffe', norm_eval=False)
    resnet50_caffe.init_weights()
    resnet50_caffe.train()
    feat = resnet50_caffe(imgs)
    assert feat.shape == torch.Size([1, 2048, 2, 2])

    resnet50_flow = ResNet(
        depth=50, pretrained='torchvision://resnet50', in_channels=10)
    input_shape = (1, 10, 64, 64)
    imgs = generate_backbone_demo_inputs(input_shape)
    feat = resnet50_flow(imgs)
    assert feat.shape == torch.Size([1, 2048, 2, 2])

    resnet50 = ResNet(
        depth=50, pretrained='torchvision://resnet50', in_channels=3)
    input_shape = (1, 3, 64, 64)
    imgs = generate_backbone_demo_inputs(input_shape)
    feat = resnet50(imgs)
    assert feat.shape == torch.Size([1, 2048, 2, 2])
