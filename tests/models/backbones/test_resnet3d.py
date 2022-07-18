# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
import torch.nn as nn
from mmcv.utils import _BatchNorm

from mmaction.models import ResNet3d, ResNet3dLayer
from ..base import check_norm_state, generate_backbone_demo_inputs


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
    imgs = generate_backbone_demo_inputs(input_shape)
    # parrots 3dconv is only implemented on gpu
    if torch.__version__ == 'parrots':
        if torch.cuda.is_available():
            resnet3d_34_frozen = resnet3d_34_frozen.cuda()
            imgs_gpu = imgs.cuda()
            feat = resnet3d_34_frozen(imgs_gpu)
            assert feat.shape == torch.Size([1, 512, 3, 2, 2])
    else:
        feat = resnet3d_34_frozen(imgs)
        assert feat.shape == torch.Size([1, 512, 3, 2, 2])

    # resnet3d with depth 50 inference
    input_shape = (1, 3, 6, 64, 64)
    imgs = generate_backbone_demo_inputs(input_shape)
    # parrots 3dconv is only implemented on gpu
    if torch.__version__ == 'parrots':
        if torch.cuda.is_available():
            resnet3d_50_frozen = resnet3d_50_frozen.cuda()
            imgs_gpu = imgs.cuda()
            feat = resnet3d_50_frozen(imgs_gpu)
            assert feat.shape == torch.Size([1, 2048, 3, 2, 2])
    else:
        feat = resnet3d_50_frozen(imgs)
        assert feat.shape == torch.Size([1, 2048, 3, 2, 2])

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
            assert feat.shape == torch.Size([1, 2048, 3, 2, 2])
    else:
        feat = resnet3d_50_caffe(imgs)
        assert feat.shape == torch.Size([1, 2048, 3, 2, 2])

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
            assert feat.shape == torch.Size([1, 512, 3, 2, 2])
    else:
        feat = resnet3d_34_caffe(imgs)
        assert feat.shape == torch.Size([1, 512, 3, 2, 2])

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
            assert feat.shape == torch.Size([1, 2048, 3, 2, 2])
    else:
        feat = resnet3d_50_1x1x1(imgs)
        assert feat.shape == torch.Size([1, 2048, 3, 2, 2])

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
            assert feat.shape == torch.Size([1, 512, 3, 2, 2])
    else:
        feat = resnet3d_34_1x1x1(imgs)
        assert feat.shape == torch.Size([1, 512, 3, 2, 2])

    # resnet3d with non-local module
    non_local_cfg = dict(
        sub_sample=True,
        use_scale=False,
        norm_cfg=dict(type='BN3d', requires_grad=True),
        mode='embedded_gaussian')
    non_local = ((0, 0, 0), (1, 0, 1, 0), (1, 0, 1, 0, 1, 0), (0, 0, 0))
    resnet3d_nonlocal = ResNet3d(
        50,
        None,
        pretrained2d=False,
        non_local=non_local,
        non_local_cfg=non_local_cfg)
    resnet3d_nonlocal.init_weights()
    for layer_name in ['layer2', 'layer3']:
        layer = getattr(resnet3d_nonlocal, layer_name)
        for i, _ in enumerate(layer):
            if i % 2 == 0:
                assert hasattr(layer[i], 'non_local_block')

    feat = resnet3d_nonlocal(imgs)
    assert feat.shape == torch.Size([1, 2048, 3, 2, 2])


def test_resnet3d_layer():
    with pytest.raises(AssertionError):
        ResNet3dLayer(22, None)

    with pytest.raises(AssertionError):
        ResNet3dLayer(50, None, stage=4)

    res_layer = ResNet3dLayer(50, None, stage=3, norm_eval=True)
    res_layer.init_weights()
    res_layer.train()
    input_shape = (1, 1024, 1, 4, 4)
    imgs = generate_backbone_demo_inputs(input_shape)
    if torch.__version__ == 'parrots':
        if torch.cuda.is_available():
            res_layer = res_layer.cuda()
            imgs_gpu = imgs.cuda()
            feat = res_layer(imgs_gpu)
            assert feat.shape == torch.Size([1, 2048, 1, 2, 2])
    else:
        feat = res_layer(imgs)
        assert feat.shape == torch.Size([1, 2048, 1, 2, 2])

    res_layer = ResNet3dLayer(
        50, 'torchvision://resnet50', stage=3, all_frozen=True)
    res_layer.init_weights()
    res_layer.train()
    imgs = generate_backbone_demo_inputs(input_shape)
    if torch.__version__ == 'parrots':
        if torch.cuda.is_available():
            res_layer = res_layer.cuda()
            imgs_gpu = imgs.cuda()
            feat = res_layer(imgs_gpu)
            assert feat.shape == torch.Size([1, 2048, 1, 2, 2])
    else:
        feat = res_layer(imgs)
        assert feat.shape == torch.Size([1, 2048, 1, 2, 2])
