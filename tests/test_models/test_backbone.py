import copy

import numpy as np
import pytest
import torch
import torch.nn as nn
from mmcv.utils import _BatchNorm

from mmaction.models import (ResNet, ResNet2Plus1d, ResNet3d, ResNet3dSlowFast,
                             ResNet3dSlowOnly, ResNetTSM)


def check_norm_state(modules, train_state):
    """Check if norm layer is in correct train state."""
    for mod in modules:
        if isinstance(mod, _BatchNorm):
            if mod.training != train_state:
                return False
    return True


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
    """Test resnet3d backbone."""
    with pytest.raises(AssertionError):
        # In ResNet3d: 1 <= num_stages <= 4
        ResNet3d(34, None, num_stages=0)

    with pytest.raises(AssertionError):
        # In ResNet3d: 1 <= num_stages <= 4
        ResNet3d(34, None, num_stages=5)

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

    with pytest.raises(AssertionError):
        # len(spatial_strides) == len(temporal_strides)
        # == len(dilations) == num_stages
        ResNet3d(
            34,
            None,
            spatial_strides=(1, ),
            temporal_strides=(1, 1),
            dilations=(1, 1, 1),
            num_stages=4)

    with pytest.raises(TypeError):
        # pretrain must be str or None.
        resnet3d_34 = ResNet3d(34, ['resnet', 'bninception'])
        resnet3d_34.init_weights()

    with pytest.raises(TypeError):
        # pretrain must be str or None.
        resnet3d_50 = ResNet3d(50, ['resnet', 'bninception'])
        resnet3d_50.init_weights()

    # resnet3d with depth 34, no pretrained, norm_eval True
    resnet3d_34 = ResNet3d(34, None, pretrained2d=False, norm_eval=True)
    resnet3d_34.init_weights()
    resnet3d_34.train()
    assert check_norm_state(resnet3d_34.modules(), False)

    # resnet3d with depth 50, no pretrained, norm_eval True
    resnet3d_50 = ResNet3d(50, None, pretrained2d=False, norm_eval=True)
    resnet3d_50.init_weights()
    resnet3d_50.train()
    assert check_norm_state(resnet3d_50.modules(), False)

    # resnet3d with depth 50, pretrained2d, norm_eval True
    resnet3d_50_pretrain = ResNet3d(
        50, 'torchvision://resnet50', norm_eval=True)
    resnet3d_50_pretrain.init_weights()
    resnet3d_50_pretrain.train()
    assert check_norm_state(resnet3d_50_pretrain.modules(), False)
    from mmcv.runner import _load_checkpoint
    chkp_2d = _load_checkpoint('torchvision://resnet50')
    for name, module in resnet3d_50_pretrain.named_modules():
        if len(name.split('.')) == 4:
            # layer.block.module.submodule
            prefix = name.split('.')[:2]
            module_type = name.split('.')[2]
            submodule_type = name.split('.')[3]

            if module_type == 'downsample':
                name2d = name.replace('conv', '0').replace('bn', '1')
            else:
                layer_id = name.split('.')[2][-1]
                name2d = prefix[0] + '.' + prefix[1] + '.' + \
                    submodule_type + layer_id

            if isinstance(module, nn.Conv3d):
                conv2d_weight = chkp_2d[name2d + '.weight']
                conv3d_weight = getattr(module, 'weight').data
                assert torch.equal(
                    conv3d_weight,
                    conv2d_weight.data.unsqueeze(2).expand_as(conv3d_weight) /
                    conv3d_weight.shape[2])
                if getattr(module, 'bias') is not None:
                    conv2d_bias = chkp_2d[name2d + '.bias']
                    conv3d_bias = getattr(module, 'bias').data
                    assert torch.equal(conv2d_bias, conv3d_bias)

            elif isinstance(module, nn.BatchNorm3d):
                for pname in ['weight', 'bias', 'running_mean', 'running_var']:
                    param_2d = chkp_2d[name2d + '.' + pname]
                    param_3d = getattr(module, pname).data
                assert torch.equal(param_2d, param_3d)

    conv3d = resnet3d_50_pretrain.conv1.conv
    assert torch.equal(
        conv3d.weight,
        chkp_2d['conv1.weight'].unsqueeze(2).expand_as(conv3d.weight) /
        conv3d.weight.shape[2])
    conv3d = resnet3d_50_pretrain.layer3[2].conv2.conv
    assert torch.equal(
        conv3d.weight, chkp_2d['layer3.2.conv2.weight'].unsqueeze(2).expand_as(
            conv3d.weight) / conv3d.weight.shape[2])

    # resnet3d with depth 34, no pretrained, norm_eval False
    resnet3d_34_no_bn_eval = ResNet3d(
        34, None, pretrained2d=False, norm_eval=False)
    resnet3d_34_no_bn_eval.init_weights()
    resnet3d_34_no_bn_eval.train()
    assert check_norm_state(resnet3d_34_no_bn_eval.modules(), True)

    # resnet3d with depth 50, no pretrained, norm_eval False
    resnet3d_50_no_bn_eval = ResNet3d(
        50, None, pretrained2d=False, norm_eval=False)
    resnet3d_50_no_bn_eval.init_weights()
    resnet3d_50_no_bn_eval.train()
    assert check_norm_state(resnet3d_50_no_bn_eval.modules(), True)

    # resnet3d with depth 34, no pretrained, frozen_stages, norm_eval False
    frozen_stages = 1
    resnet3d_34_frozen = ResNet3d(
        34, None, pretrained2d=False, frozen_stages=frozen_stages)
    resnet3d_34_frozen.init_weights()
    resnet3d_34_frozen.train()
    assert resnet3d_34_frozen.conv1.bn.training is False
    for param in resnet3d_34_frozen.conv1.parameters():
        assert param.requires_grad is False
    for i in range(1, frozen_stages + 1):
        layer = getattr(resnet3d_34_frozen, f'layer{i}')
        for mod in layer.modules():
            if isinstance(mod, _BatchNorm):
                assert mod.training is False
        for param in layer.parameters():
            assert param.requires_grad is False
    # test zero_init_residual
    for m in resnet3d_34_frozen.modules():
        if hasattr(m, 'conv2'):
            assert torch.equal(m.conv2.bn.weight,
                               torch.zeros_like(m.conv2.bn.weight))
            assert torch.equal(m.conv2.bn.bias,
                               torch.zeros_like(m.conv2.bn.bias))

    # resnet3d with depth 50, no pretrained, frozen_stages, norm_eval False
    frozen_stages = 1
    resnet3d_50_frozen = ResNet3d(
        50, None, pretrained2d=False, frozen_stages=frozen_stages)
    resnet3d_50_frozen.init_weights()
    resnet3d_50_frozen.train()
    assert resnet3d_50_frozen.conv1.bn.training is False
    for param in resnet3d_50_frozen.conv1.parameters():
        assert param.requires_grad is False
    for i in range(1, frozen_stages + 1):
        layer = getattr(resnet3d_50_frozen, f'layer{i}')
        for mod in layer.modules():
            if isinstance(mod, _BatchNorm):
                assert mod.training is False
        for param in layer.parameters():
            assert param.requires_grad is False
    # test zero_init_residual
    for m in resnet3d_50_frozen.modules():
        if hasattr(m, 'conv3'):
            assert torch.equal(m.conv3.bn.weight,
                               torch.zeros_like(m.conv3.bn.weight))
            assert torch.equal(m.conv3.bn.bias,
                               torch.zeros_like(m.conv3.bn.bias))

    # resnet3d frozen with depth 34 inference
    input_shape = (1, 3, 6, 64, 64)
    imgs = _demo_inputs(input_shape)
    # parrots 3dconv is only implemented on gpu
    if torch.__version__ == 'parrots':
        if torch.cuda.is_available():
            resnet3d_34_frozen = resnet3d_34_frozen.cuda()
            imgs_gpu = imgs.cuda()
            feat = resnet3d_34_frozen(imgs_gpu)
            assert feat.shape == torch.Size([1, 512, 1, 2, 2])
    else:
        feat = resnet3d_34_frozen(imgs)
        assert feat.shape == torch.Size([1, 512, 1, 2, 2])

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

    # resnet3d with depth 34 in caffe style inference
    resnet3d_34_caffe = ResNet3d(34, None, pretrained2d=False, style='caffe')
    resnet3d_34_caffe.init_weights()
    resnet3d_34_caffe.train()
    # parrots 3dconv is only implemented on gpu
    if torch.__version__ == 'parrots':
        if torch.cuda.is_available():
            resnet3d_34_caffe = resnet3d_34_caffe.cuda()
            imgs_gpu = imgs.cuda()
            feat = resnet3d_34_caffe(imgs_gpu)
            assert feat.shape == torch.Size([1, 512, 1, 2, 2])
    else:
        feat = resnet3d_34_caffe(imgs)
        assert feat.shape == torch.Size([1, 512, 1, 2, 2])

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

    resnet3d_34_1x1x1 = ResNet3d(
        34, None, pretrained2d=False, inflate_style='3x3x3')
    resnet3d_34_1x1x1.init_weights()
    resnet3d_34_1x1x1.train()

    # parrots 3dconv is only implemented on gpu
    if torch.__version__ == 'parrots':
        if torch.cuda.is_available():
            resnet3d_34_1x1x1 = resnet3d_34_1x1x1.cuda()
            imgs_gpu = imgs.cuda()
            feat = resnet3d_34_1x1x1(imgs_gpu)
            assert feat.shape == torch.Size([1, 512, 1, 2, 2])
    else:
        feat = resnet3d_34_1x1x1(imgs)
        assert feat.shape == torch.Size([1, 512, 1, 2, 2])


def test_resnet2plus1d_backbone():
    # Test r2+1d backbone
    with pytest.raises(AssertionError):
        # r2+1d does not support inflation
        ResNet2Plus1d(50, None, pretrained2d=True)

    with pytest.raises(AssertionError):
        # r2+1d requires conv(2+1)d module
        ResNet2Plus1d(
            50, None, pretrained2d=False, conv_cfg=dict(type='Conv3d'))

    frozen_stages = 1
    r2plus1d_34_frozen = ResNet2Plus1d(
        34,
        None,
        conv_cfg=dict(type='Conv2plus1d'),
        pretrained2d=False,
        frozen_stages=frozen_stages,
        conv1_kernel=(3, 7, 7),
        conv1_stride_t=1,
        pool1_stride_t=1,
        inflate=(1, 1, 1, 1),
        spatial_strides=(1, 2, 2, 2),
        temporal_strides=(1, 2, 2, 2))
    r2plus1d_34_frozen.init_weights()
    r2plus1d_34_frozen.train()
    assert r2plus1d_34_frozen.conv1.conv.bn_s.training is False
    assert r2plus1d_34_frozen.conv1.bn.training is False
    for param in r2plus1d_34_frozen.conv1.parameters():
        assert param.requires_grad is False
    for i in range(1, frozen_stages + 1):
        layer = getattr(r2plus1d_34_frozen, f'layer{i}')
        for mod in layer.modules():
            if isinstance(mod, _BatchNorm):
                assert mod.training is False
        for param in layer.parameters():
            assert param.requires_grad is False
    input_shape = (1, 3, 8, 64, 64)
    imgs = _demo_inputs(input_shape)
    # parrots 3dconv is only implemented on gpu
    if torch.__version__ == 'parrots':
        if torch.cuda.is_available():
            r2plus1d_34_frozen = r2plus1d_34_frozen.cuda()
            imgs_gpu = imgs.cuda()
            feat = r2plus1d_34_frozen(imgs_gpu)
            assert feat.shape == torch.Size([1, 512, 1, 2, 2])
    else:
        feat = r2plus1d_34_frozen(imgs)
        assert feat.shape == torch.Size([1, 512, 1, 2, 2])

    r2plus1d_50_frozen = ResNet2Plus1d(
        50,
        None,
        conv_cfg=dict(type='Conv2plus1d'),
        pretrained2d=False,
        conv1_kernel=(3, 7, 7),
        conv1_stride_t=1,
        pool1_stride_t=1,
        inflate=(1, 1, 1, 1),
        spatial_strides=(1, 2, 2, 2),
        temporal_strides=(1, 2, 2, 2),
        frozen_stages=frozen_stages)
    r2plus1d_50_frozen.init_weights()

    r2plus1d_50_frozen.train()
    assert r2plus1d_50_frozen.conv1.conv.bn_s.training is False
    assert r2plus1d_50_frozen.conv1.bn.training is False
    for param in r2plus1d_50_frozen.conv1.parameters():
        assert param.requires_grad is False
    for i in range(1, frozen_stages + 1):
        layer = getattr(r2plus1d_50_frozen, f'layer{i}')
        for mod in layer.modules():
            if isinstance(mod, _BatchNorm):
                assert mod.training is False
        for param in layer.parameters():
            assert param.requires_grad is False
    input_shape = (1, 3, 8, 64, 64)
    imgs = _demo_inputs(input_shape)

    # parrots 3dconv is only implemented on gpu
    if torch.__version__ == 'parrots':
        if torch.cuda.is_available():
            r2plus1d_50_frozen = r2plus1d_50_frozen.cuda()
            imgs_gpu = imgs.cuda()
            feat = r2plus1d_50_frozen(imgs_gpu)
            assert feat.shape == torch.Size([1, 2048, 1, 2, 2])
    else:
        feat = r2plus1d_50_frozen(imgs)
        assert feat.shape == torch.Size([1, 2048, 1, 2, 2])


def test_resnet_tsm_backbone():
    """Test resnet_tsm backbone."""
    with pytest.raises(NotImplementedError):
        # shift_place must be block or blockres
        resnet_tsm_50_block = ResNetTSM(50, shift_place='Block')
        resnet_tsm_50_block.init_weights()

    from mmaction.models.backbones.resnet_tsm import TemporalShift
    from mmaction.models.backbones.resnet import Bottleneck

    # resnet_tsm with depth 50
    resnet_tsm_50 = ResNetTSM(50)
    resnet_tsm_50.init_weights()
    for layer_name in resnet_tsm_50.res_layers:
        layer = getattr(resnet_tsm_50, layer_name)
        blocks = list(layer.children())
        for block in blocks:
            assert isinstance(block.conv1.conv, TemporalShift)
            assert block.conv1.conv.num_segments == resnet_tsm_50.num_segments
            assert block.conv1.conv.shift_div == resnet_tsm_50.shift_div
            assert isinstance(block.conv1.conv.net, nn.Conv2d)

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
            assert block.shift_div == resnet_tsm_50_block.shift_div
            assert isinstance(block.net, Bottleneck)

    # resnet_tsm with depth 50, no pretrained, use temporal_pool
    resnet_tsm_50_temporal_pool = ResNetTSM(50, temporal_pool=True)
    resnet_tsm_50_temporal_pool.init_weights()
    for layer_name in resnet_tsm_50_temporal_pool.res_layers:
        layer = getattr(resnet_tsm_50_temporal_pool, layer_name)
        blocks = list(layer.children())

        if layer_name == 'layer2':
            assert len(blocks) == 2
            assert isinstance(blocks[1], nn.MaxPool3d)
            blocks = copy.deepcopy(blocks[0])

        for block in blocks:
            assert isinstance(block.conv1.conv, TemporalShift)
            if layer_name == 'layer1':
                assert block.conv1.conv.num_segments == \
                       resnet_tsm_50_temporal_pool.num_segments
            else:
                assert block.conv1.conv.num_segments == \
                       resnet_tsm_50_temporal_pool.num_segments // 2
            assert block.conv1.conv.shift_div == resnet_tsm_50_temporal_pool.shift_div  # noqa: E501
            assert isinstance(block.conv1.conv.net, nn.Conv2d)

    input_shape = (8, 3, 64, 64)
    imgs = _demo_inputs(input_shape)

    feat = resnet_tsm_50(imgs)
    assert feat.shape == torch.Size([8, 2048, 2, 2])

    feat = resnet_tsm_50_temporal_pool(imgs)
    assert feat.shape == torch.Size([4, 2048, 2, 2])


def test_slowfast_backbone():
    """Test SlowFast backbone."""
    with pytest.raises(TypeError):
        # cfg should be a dict
        ResNet3dSlowFast(None, slow_pathway=list(['foo', 'bar']))
    with pytest.raises(TypeError):
        # pretrained should be a str
        sf_50 = ResNet3dSlowFast(dict(foo='bar'))
        sf_50.init_weights()
    with pytest.raises(KeyError):
        # pathway type should be implemented
        ResNet3dSlowFast(None, slow_pathway=dict(type='resnext'))

    # test slowfast with slow inflated
    sf_50_inflate = ResNet3dSlowFast(
        None,
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
    imgs = _demo_inputs(input_shape)
    # parrots 3dconv is only implemented on gpu
    if torch.__version__ == 'parrots':
        if torch.cuda.is_available():
            sf_50_wo_lateral = sf_50_wo_lateral.cuda()
            imgs_gpu = imgs.cuda()
            feat = sf_50_wo_lateral(imgs_gpu)
    else:
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
    sf_50 = ResNet3dSlowFast(None)
    sf_50.init_weights()
    sf_50.train()

    # slowfast inference test
    input_shape = (1, 3, 8, 64, 64)
    imgs = _demo_inputs(input_shape)
    # parrots 3dconv is only implemented on gpu
    if torch.__version__ == 'parrots':
        if torch.cuda.is_available():
            sf_50 = sf_50.cuda()
            imgs_gpu = imgs.cuda()
            feat = sf_50(imgs_gpu)
    else:
        feat = sf_50(imgs)

    assert isinstance(feat, tuple)
    assert feat[0].shape == torch.Size([1, 2048, 1, 2, 2])
    assert feat[1].shape == torch.Size([1, 256, 8, 2, 2])


def test_slowonly_backbone():
    """Test SlowOnly backbone."""
    with pytest.raises(AssertionError):
        # SlowOnly should contain no lateral connection
        ResNet3dSlowOnly(50, None, lateral=True)

    # test SlowOnly with normal config
    so_50 = ResNet3dSlowOnly(50, None)
    so_50.init_weights()
    so_50.train()

    # SlowOnly inference test
    input_shape = (1, 3, 8, 64, 64)
    imgs = _demo_inputs(input_shape)
    # parrots 3dconv is only implemented on gpu
    if torch.__version__ == 'parrots':
        if torch.cuda.is_available():
            so_50 = so_50.cuda()
            imgs_gpu = imgs.cuda()
            feat = so_50(imgs_gpu)
    else:
        feat = so_50(imgs)
    assert feat.shape == torch.Size([1, 2048, 8, 2, 2])


def _demo_inputs(input_shape=(1, 3, 64, 64)):
    """Create a superset of inputs needed to run backbone.

    Args:
        input_shape (tuple): input batch dimensions.
            Default: (1, 3, 64, 64).
    """
    imgs = np.random.random(input_shape)
    imgs = torch.FloatTensor(imgs)

    return imgs
