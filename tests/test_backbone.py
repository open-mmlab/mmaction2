import numpy as np
import pytest
import torch
import torch.nn as nn

from mmaction.models import ResNet, ResNet3d, ResNetTIN, ResNetTSM
from mmaction.utils import _BatchNorm


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
    # parrots 3dconv is only implemented on gpu
    if torch.__version__ == 'parrots':
        if torch.cuda.is_available():
            resnet3d_50_frozen = resnet3d_50_frozen.cuda()
            imgs_gpu = imgs.cuda()
            feat = resnet3d_50_frozen(imgs_gpu)
            assert feat.shape == torch.Size([1, 2048, 1, 2, 2])
    else:
        feat = resnet3d_50_frozen(imgs)
        assert feat.shape == torch.Size([1, 2048, 1, 2, 2])

    # resnet3d with depth 50 in caffe style inference
    resnet3d_50_caffe = ResNet3d(50, None, pretrained2d=False, style='caffe')
    resnet3d_50_caffe.init_weights()
    resnet3d_50_caffe.train()
    # parrots 3dconv is only implemented on gpu
    if torch.__version__ == 'parrots':
        if torch.cuda.is_available():
            resnet3d_50_caffe = resnet3d_50_caffe.cuda()
            imgs_gpu = imgs.cuda()
            feat = resnet3d_50_caffe(imgs_gpu)
            assert feat.shape == torch.Size([1, 2048, 1, 2, 2])
    else:
        feat = resnet3d_50_caffe(imgs)
        assert feat.shape == torch.Size([1, 2048, 1, 2, 2])

    # resnet3d with depth with 3x3x3 inflate_style inference
    resnet3d_50_1x1x1 = ResNet3d(
        50, None, pretrained2d=False, inflate_style='3x3x3')
    resnet3d_50_1x1x1.init_weights()
    resnet3d_50_1x1x1.train()
    # parrots 3dconv is only implemented on gpu
    if torch.__version__ == 'parrots':
        if torch.cuda.is_available():
            resnet3d_50_1x1x1 = resnet3d_50_1x1x1.cuda()
            imgs_gpu = imgs.cuda()
            feat = resnet3d_50_1x1x1(imgs_gpu)
            assert feat.shape == torch.Size([1, 2048, 1, 2, 2])
    else:
        feat = resnet3d_50_1x1x1(imgs)
        assert feat.shape == torch.Size([1, 2048, 1, 2, 2])


def test_resnet_tsm_backbone():
    """Test resnet_tsm backbone"""
    with pytest.raises(NotImplementedError):
        # shift_place must be block or blockres
        resnet_tsm_50_Block = ResNetTSM(50, shift_place='Block')
        resnet_tsm_50_Block.init_weights()

    from mmaction.models.backbones.resnet_tsm import TemporalShift
    from mmaction.models.backbones.resnet import Bottleneck

    # resnet_tsm with depth 50
    resnet_tsm_50 = ResNetTSM(50)
    resnet_tsm_50.init_weights()
    for layer_name in resnet_tsm_50.res_layers:
        layer = getattr(resnet_tsm_50, layer_name)
        blocks = list(layer.children())
        for block in blocks:
            assert isinstance(block.conv1, TemporalShift)
            assert block.conv1.num_segments == resnet_tsm_50.num_segments
            assert block.conv1.fold_div == resnet_tsm_50.shift_div
            assert isinstance(block.conv1.net, nn.Conv2d)

    # resnet_tsm with depth 50, no pretrained, shift_place is block
    resnet_tsm_50_block = ResNetTSM(50, shift_place='block')
    resnet_tsm_50_block.init_weights()
    for layer_name in resnet_tsm_50_block.res_layers:
        layer = getattr(resnet_tsm_50_block, layer_name)
        blocks = list(layer.children())
        for block in blocks:
            assert isinstance(block, TemporalShift)
            assert block.num_segments == resnet_tsm_50_block.num_segments
            assert block.num_segments == resnet_tsm_50_block.num_segments
            assert block.fold_div == resnet_tsm_50_block.shift_div
            assert isinstance(block.net, Bottleneck)

    # resnet_tsm with depth 50, no pretrained, use temporal_pool
    resnet_tsm_50_temporal_pool = ResNetTSM(50, temporal_pool=True)
    resnet_tsm_50_temporal_pool.init_weights()
    for layer_name in resnet_tsm_50_temporal_pool.res_layers:
        layer = getattr(resnet_tsm_50_temporal_pool, layer_name)
        blocks = list(layer.children())
        for block in blocks:
            assert isinstance(block.conv1, TemporalShift)
            if layer_name == 'layer1':
                assert block.conv1.num_segments == \
                       resnet_tsm_50_temporal_pool.num_segments
            else:
                assert block.conv1.num_segments == \
                       resnet_tsm_50_temporal_pool.num_segments // 2
            assert block.conv1.fold_div == resnet_tsm_50_temporal_pool.shift_div  # noqa: E501
            assert isinstance(block.conv1.net, nn.Conv2d)

    # compare ResNetTSM with ResNet when using pretrained.
    resnet_tsm_50_no_shift = ResNetTSM(
        50, is_shift=False, pretrained='torchvision://resnet50')
    resnet_tsm_50_no_shift.init_weights()
    resnet_tsm_dict = {
        k: v
        for k, v in resnet_tsm_50_no_shift.named_parameters()
    }

    resnet = ResNet(50, pretrained='torchvision://resnet50')
    resnet.init_weights()
    resnet_dict = {k: v for k, v in resnet.named_parameters()}

    assert set(resnet_tsm_dict.keys()) == set(resnet_dict.keys())
    for k in resnet_tsm_dict.keys():
        assert torch.equal(resnet_tsm_dict[k], resnet_dict[k])


def test_resnet_tin_backbone():
    """Test resnet_tin backbone"""
    with pytest.raises(TypeError):
        # finetune must be a str or None
        resnet_tin = ResNetTIN(50, finetune=0)
        resnet_tin.init_weights()

    with pytest.raises(AssertionError):
        # num_segments should be positive
        resnet_tin = ResNetTIN(50, num_segments=-1)
        resnet_tin.init_weights()

    from mmaction.models.backbones.resnet_tin import \
        TemporalInterlace, CombineNet

    # resnet_tin with normal config
    resnet_tin = ResNetTIN(50)
    resnet_tin.init_weights()
    for layer_name in resnet_tin.res_layers:
        layer = getattr(resnet_tin, layer_name)
        blocks = list(layer.children())
        for block in blocks:
            assert isinstance(block.conv1, CombineNet)
            assert isinstance(block.conv1.net1, TemporalInterlace)
            assert isinstance(block.conv1.net2, nn.Conv2d)
            assert block.conv1.net1.num_segments == resnet_tin.num_segments
            assert block.conv1.net1.shift_div == resnet_tin.shift_div

    input_shape = (8, 3, 64, 64)
    imgs = _demo_inputs(input_shape)

    # resnet_tin with normal cfg inference
    feat = resnet_tin(imgs)
    assert feat.shape == torch.Size([8, 2048, 2, 2])


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
