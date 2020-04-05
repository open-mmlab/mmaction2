import numpy as np
import pytest
import torch
from torch.nn.modules.batchnorm import _BatchNorm

from mmaction.models import ResNet, ResNet3d


def check_norm_state(modules, train_state):
    """Check if norm layer is in correct train state."""
    for mod in modules:
        if isinstance(mod, _BatchNorm):
            if mod.training != train_state:
                return False
    return True


def test_resnet_backbone():
    """Test resnet backbone"""
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
    assert resnet50_frozen.norm1.training is False
    for layer in [resnet50_frozen.conv1, resnet50_frozen.norm1]:
        for param in layer.parameters():
            assert param.requires_grad is False
    for i in range(1, frozen_stages + 1):
        layer = getattr(resnet50_frozen, f'layer{i}')
        for mod in layer.modules():
            if isinstance(mod, _BatchNorm):
                assert mod.training is False
        for param in layer.parameters():
            assert param.requires_grad is False

    input_shape = (1, 3, 64, 64)
    imgs = _demo_inputs(input_shape)

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


def test_resnet3d_backbone():
    """Test resnet3d backbone"""
    with pytest.raises(KeyError):
        # depth must in [50, 101, 152]
        ResNet3d(18, None)

    with pytest.raises(AssertionError):
        # In ResNet3d: 1 <= num_stages <= 4
        ResNet3d(50, None, num_stages=0)

    with pytest.raises(AssertionError):
        # In ResNet3d: 1 <= num_stages <= 4
        ResNet3d(50, None, num_stages=5)

    with pytest.raises(AssertionError):
        # len(spatial_strides) == len(temporal_strides)
        # == len(dilations) == num_stages
        ResNet3d(
            50,
            None,
            spatial_strides=(1, ),
            temporal_strides=(1, 1),
            dilations=(1, 1, 1),
            num_stages=4)

    with pytest.raises(TypeError):
        # pretrain must be str or None.
        resnet3d_50 = ResNet3d(50, ['resnet', 'bninception'])
        resnet3d_50.init_weights()

    # resnet3d with depth 50, no pretrained, norm_eval True
    resnet3d_50 = ResNet3d(50, None, pretrained2d=False, norm_eval=True)
    resnet3d_50.init_weights()
    resnet3d_50.train()
    assert check_norm_state(resnet3d_50.modules(), False)

    # resnet3d with depth 50, pretrained, norm_eval True
    resnet3d_50_pretrain = ResNet3d(
        50, 'torchvision://resnet50', norm_eval=True)
    resnet3d_50_pretrain.init_weights()
    resnet3d_50_pretrain.train()
    assert check_norm_state(resnet3d_50_pretrain.modules(), False)

    # resnet3d with depth 50, no pretrained, norm_eval False
    resnet3d_50_no_bn_eval = ResNet3d(
        50, None, pretrained2d=False, norm_eval=False)
    resnet3d_50_no_bn_eval.init_weights()
    resnet3d_50_no_bn_eval.train()
    assert check_norm_state(resnet3d_50_no_bn_eval.modules(), True)

    # resnet3d with depth 50, no pretrained, frozen_stages, norm_eval False
    frozen_stages = 1
    resnet3d_50_frozen = ResNet3d(
        50, None, pretrained2d=False, frozen_stages=frozen_stages)
    resnet3d_50_frozen.init_weights()
    resnet3d_50_frozen.train()
    assert resnet3d_50_frozen.norm1.training is False
    for layer in [resnet3d_50_frozen.conv1, resnet3d_50_frozen.norm1]:
        for param in layer.parameters():
            assert param.requires_grad is False
    for i in range(1, frozen_stages + 1):
        layer = getattr(resnet3d_50_frozen, f'layer{i}')
        for mod in layer.modules():
            if isinstance(mod, _BatchNorm):
                assert mod.training is False
        for param in layer.parameters():
            assert param.requires_grad is False

    # resnet3d with depth 50 inference
    input_shape = (1, 3, 6, 64, 64)
    imgs = _demo_inputs(input_shape)
    feat = resnet3d_50_frozen(imgs)
    assert feat.shape == torch.Size([1, 2048, 1, 2, 2])

    # resnet3d with depth 50 in caffe style inference
    resnet3d_50_caffe = ResNet3d(50, None, pretrained2d=False, style='caffe')
    resnet3d_50_caffe.init_weights()
    resnet3d_50_caffe.train()
    feat = resnet3d_50_caffe(imgs)
    assert feat.shape == torch.Size([1, 2048, 1, 2, 2])

    resnet3d_50_1x1x1 = ResNet3d(
        50, None, pretrained2d=False, inflate_style='3x3x3')
    resnet3d_50_1x1x1.init_weights()
    resnet3d_50_1x1x1.train()
    feat = resnet3d_50_1x1x1(imgs)
    assert feat.shape == torch.Size([1, 2048, 1, 2, 2])


def _demo_inputs(input_shape=(1, 3, 64, 64)):
    """
    Create a superset of inputs needed to run backbone.

    Args:
        input_shape (tuple): input batch dimensions.
            Default: (1, 3, 64, 64).
    """
    imgs = np.random.random(input_shape)
    imgs = torch.FloatTensor(imgs)

    return imgs
