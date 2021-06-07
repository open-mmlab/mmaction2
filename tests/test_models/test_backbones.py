import copy

import pytest
import torch
import torch.nn as nn
from mmcv.utils import _BatchNorm

from mmaction.models import (C3D, X3D, MobileNetV2TSM, ResNet2Plus1d,
                             ResNet3dCSN, ResNet3dSlowFast, ResNet3dSlowOnly,
                             ResNetAudio, ResNetTIN, ResNetTSM, TANet)
from mmaction.models.backbones.resnet_tsm import NL3DWrapper
from .base import check_norm_state, generate_backbone_demo_inputs


def test_x3d_backbone():
    """Test x3d backbone."""
    with pytest.raises(AssertionError):
        # In X3D: 1 <= num_stages <= 4
        X3D(gamma_w=1.0, gamma_b=2.25, gamma_d=2.2, num_stages=0)

    with pytest.raises(AssertionError):
        # In X3D: 1 <= num_stages <= 4
        X3D(gamma_w=1.0, gamma_b=2.25, gamma_d=2.2, num_stages=5)

    with pytest.raises(AssertionError):
        # len(spatial_strides) == num_stages
        X3D(gamma_w=1.0,
            gamma_b=2.25,
            gamma_d=2.2,
            spatial_strides=(1, 2),
            num_stages=4)

    with pytest.raises(AssertionError):
        # se_style in ['half', 'all']
        X3D(gamma_w=1.0, gamma_b=2.25, gamma_d=2.2, se_style=None)

    with pytest.raises(AssertionError):
        # se_ratio should be None or > 0
        X3D(gamma_w=1.0,
            gamma_b=2.25,
            gamma_d=2.2,
            se_style='half',
            se_ratio=0)

    # x3d_s, no pretrained, norm_eval True
    x3d_s = X3D(gamma_w=1.0, gamma_b=2.25, gamma_d=2.2, norm_eval=True)
    x3d_s.init_weights()
    x3d_s.train()
    assert check_norm_state(x3d_s.modules(), False)

    # x3d_l, no pretrained, norm_eval True
    x3d_l = X3D(gamma_w=1.0, gamma_b=2.25, gamma_d=5.0, norm_eval=True)
    x3d_l.init_weights()
    x3d_l.train()
    assert check_norm_state(x3d_l.modules(), False)

    # x3d_s, no pretrained, norm_eval False
    x3d_s = X3D(gamma_w=1.0, gamma_b=2.25, gamma_d=2.2, norm_eval=False)
    x3d_s.init_weights()
    x3d_s.train()
    assert check_norm_state(x3d_s.modules(), True)

    # x3d_l, no pretrained, norm_eval False
    x3d_l = X3D(gamma_w=1.0, gamma_b=2.25, gamma_d=5.0, norm_eval=False)
    x3d_l.init_weights()
    x3d_l.train()
    assert check_norm_state(x3d_l.modules(), True)

    # x3d_s, no pretrained, frozen_stages, norm_eval False
    frozen_stages = 1
    x3d_s_frozen = X3D(
        gamma_w=1.0,
        gamma_b=2.25,
        gamma_d=2.2,
        norm_eval=False,
        frozen_stages=frozen_stages)

    x3d_s_frozen.init_weights()
    x3d_s_frozen.train()
    assert x3d_s_frozen.conv1_t.bn.training is False
    for param in x3d_s_frozen.conv1_s.parameters():
        assert param.requires_grad is False
    for param in x3d_s_frozen.conv1_t.parameters():
        assert param.requires_grad is False

    for i in range(1, frozen_stages + 1):
        layer = getattr(x3d_s_frozen, f'layer{i}')
        for mod in layer.modules():
            if isinstance(mod, _BatchNorm):
                assert mod.training is False
        for param in layer.parameters():
            assert param.requires_grad is False

    # test zero_init_residual, zero_init_residual is True by default
    for m in x3d_s_frozen.modules():
        if hasattr(m, 'conv3'):
            assert torch.equal(m.conv3.bn.weight,
                               torch.zeros_like(m.conv3.bn.weight))
            assert torch.equal(m.conv3.bn.bias,
                               torch.zeros_like(m.conv3.bn.bias))

    # x3d_s inference
    input_shape = (1, 3, 13, 64, 64)
    imgs = generate_backbone_demo_inputs(input_shape)
    # parrots 3dconv is only implemented on gpu
    if torch.__version__ == 'parrots':
        if torch.cuda.is_available():
            x3d_s_frozen = x3d_s_frozen.cuda()
            imgs_gpu = imgs.cuda()
            feat = x3d_s_frozen(imgs_gpu)
            assert feat.shape == torch.Size([1, 432, 13, 2, 2])
    else:
        feat = x3d_s_frozen(imgs)
        assert feat.shape == torch.Size([1, 432, 13, 2, 2])

    # x3d_m inference
    input_shape = (1, 3, 16, 96, 96)
    imgs = generate_backbone_demo_inputs(input_shape)
    # parrots 3dconv is only implemented on gpu
    if torch.__version__ == 'parrots':
        if torch.cuda.is_available():
            x3d_s_frozen = x3d_s_frozen.cuda()
            imgs_gpu = imgs.cuda()
            feat = x3d_s_frozen(imgs_gpu)
            assert feat.shape == torch.Size([1, 432, 16, 3, 3])
    else:
        feat = x3d_s_frozen(imgs)
        assert feat.shape == torch.Size([1, 432, 16, 3, 3])


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
    imgs = generate_backbone_demo_inputs(input_shape)
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
    imgs = generate_backbone_demo_inputs(input_shape)

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

    from mmaction.models.backbones.resnet import Bottleneck
    from mmaction.models.backbones.resnet_tsm import TemporalShift

    input_shape = (8, 3, 64, 64)
    imgs = generate_backbone_demo_inputs(input_shape)

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

    # resnet_tsm with non-local module
    non_local_cfg = dict(
        sub_sample=True,
        use_scale=False,
        norm_cfg=dict(type='BN3d', requires_grad=True),
        mode='embedded_gaussian')
    non_local = ((0, 0, 0), (1, 0, 1, 0), (1, 0, 1, 0, 1, 0), (0, 0, 0))
    resnet_tsm_nonlocal = ResNetTSM(
        50, non_local=non_local, non_local_cfg=non_local_cfg)
    resnet_tsm_nonlocal.init_weights()
    for layer_name in ['layer2', 'layer3']:
        layer = getattr(resnet_tsm_nonlocal, layer_name)
        for i, _ in enumerate(layer):
            if i % 2 == 0:
                assert isinstance(layer[i], NL3DWrapper)

    resnet_tsm_50_full = ResNetTSM(
        50,
        non_local=non_local,
        non_local_cfg=non_local_cfg,
        temporal_pool=True)
    resnet_tsm_50_full.init_weights()

    # TSM forword
    feat = resnet_tsm_50(imgs)
    assert feat.shape == torch.Size([8, 2048, 2, 2])

    # TSM with non-local forward
    feat = resnet_tsm_nonlocal(imgs)
    assert feat.shape == torch.Size([8, 2048, 2, 2])

    # TSM with temporal pool forward
    feat = resnet_tsm_50_temporal_pool(imgs)
    assert feat.shape == torch.Size([4, 2048, 2, 2])

    # TSM with temporal pool + non-local forward
    input_shape = (16, 3, 32, 32)
    imgs = generate_backbone_demo_inputs(input_shape)
    feat = resnet_tsm_50_full(imgs)
    assert feat.shape == torch.Size([8, 2048, 1, 1])


def test_mobilenetv2_tsm_backbone():
    """Test mobilenetv2_tsm backbone."""
    from mmaction.models.backbones.resnet_tsm import TemporalShift
    from mmaction.models.backbones.mobilenet_v2 import InvertedResidual
    from mmcv.cnn import ConvModule

    input_shape = (8, 3, 64, 64)
    imgs = generate_backbone_demo_inputs(input_shape)

    # mobilenetv2_tsm with width_mult = 1.0
    mobilenetv2_tsm = MobileNetV2TSM()
    mobilenetv2_tsm.init_weights()
    for cur_module in mobilenetv2_tsm.modules():
        if isinstance(cur_module, InvertedResidual) and \
            len(cur_module.conv) == 3 and \
                cur_module.use_res_connect:
            assert isinstance(cur_module.conv[0], TemporalShift)
            assert cur_module.conv[0].num_segments == \
                mobilenetv2_tsm.num_segments
            assert cur_module.conv[0].shift_div == mobilenetv2_tsm.shift_div
            assert isinstance(cur_module.conv[0].net, ConvModule)

    # TSM-MobileNetV2 with widen_factor = 1.0 forword
    feat = mobilenetv2_tsm(imgs)
    assert feat.shape == torch.Size([8, 1280, 2, 2])

    # mobilenetv2 with widen_factor = 0.5 forword
    mobilenetv2_tsm_05 = MobileNetV2TSM(widen_factor=0.5)
    mobilenetv2_tsm_05.init_weights()
    feat = mobilenetv2_tsm_05(imgs)
    assert feat.shape == torch.Size([8, 1280, 2, 2])

    # mobilenetv2 with widen_factor = 1.5 forword
    mobilenetv2_tsm_15 = MobileNetV2TSM(widen_factor=1.5)
    mobilenetv2_tsm_15.init_weights()
    feat = mobilenetv2_tsm_15(imgs)
    assert feat.shape == torch.Size([8, 1920, 2, 2])


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
    imgs = generate_backbone_demo_inputs(input_shape)
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
    imgs = generate_backbone_demo_inputs(input_shape)
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

    # test SlowOnly for PoseC3D
    so_50 = ResNet3dSlowOnly(
        depth=50,
        pretrained=None,
        in_channels=17,
        base_channels=32,
        num_stages=3,
        out_indices=(2, ),
        stage_blocks=(4, 6, 3),
        conv1_stride_s=1,
        pool1_stride_s=1,
        inflate=(0, 1, 1),
        spatial_strides=(2, 2, 2),
        temporal_strides=(1, 1, 2),
        dilations=(1, 1, 1))
    so_50.init_weights()
    so_50.train()

    # test SlowOnly with normal config
    so_50 = ResNet3dSlowOnly(50, None)
    so_50.init_weights()
    so_50.train()

    # SlowOnly inference test
    input_shape = (1, 3, 8, 64, 64)
    imgs = generate_backbone_demo_inputs(input_shape)
    # parrots 3dconv is only implemented on gpu
    if torch.__version__ == 'parrots':
        if torch.cuda.is_available():
            so_50 = so_50.cuda()
            imgs_gpu = imgs.cuda()
            feat = so_50(imgs_gpu)
    else:
        feat = so_50(imgs)
    assert feat.shape == torch.Size([1, 2048, 8, 2, 2])


def test_resnet_csn_backbone():
    """Test resnet_csn backbone."""
    with pytest.raises(ValueError):
        # Bottleneck mode must be "ip" or "ir"
        ResNet3dCSN(152, None, bottleneck_mode='id')

    input_shape = (2, 3, 6, 64, 64)
    imgs = generate_backbone_demo_inputs(input_shape)

    resnet3d_csn_frozen = ResNet3dCSN(
        152, None, bn_frozen=True, norm_eval=True)
    resnet3d_csn_frozen.train()
    for m in resnet3d_csn_frozen.modules():
        if isinstance(m, _BatchNorm):
            for param in m.parameters():
                assert param.requires_grad is False

    # Interaction-preserved channel-separated bottleneck block
    resnet3d_csn_ip = ResNet3dCSN(152, None, bottleneck_mode='ip')
    resnet3d_csn_ip.init_weights()
    resnet3d_csn_ip.train()
    for i, layer_name in enumerate(resnet3d_csn_ip.res_layers):
        layers = getattr(resnet3d_csn_ip, layer_name)
        num_blocks = resnet3d_csn_ip.stage_blocks[i]
        assert len(layers) == num_blocks
        for layer in layers:
            assert isinstance(layer.conv2, nn.Sequential)
            assert len(layer.conv2) == 2
            assert layer.conv2[1].groups == layer.planes
    if torch.__version__ == 'parrots':
        if torch.cuda.is_available():
            resnet3d_csn_ip = resnet3d_csn_ip.cuda()
            imgs_gpu = imgs.cuda()
            feat = resnet3d_csn_ip(imgs_gpu)
            assert feat.shape == torch.Size([2, 2048, 1, 2, 2])
    else:
        feat = resnet3d_csn_ip(imgs)
        assert feat.shape == torch.Size([2, 2048, 1, 2, 2])

    # Interaction-reduced channel-separated bottleneck block
    resnet3d_csn_ir = ResNet3dCSN(152, None, bottleneck_mode='ir')
    resnet3d_csn_ir.init_weights()
    resnet3d_csn_ir.train()
    for i, layer_name in enumerate(resnet3d_csn_ir.res_layers):
        layers = getattr(resnet3d_csn_ir, layer_name)
        num_blocks = resnet3d_csn_ir.stage_blocks[i]
        assert len(layers) == num_blocks
        for layer in layers:
            assert isinstance(layer.conv2, nn.Sequential)
            assert len(layer.conv2) == 1
            assert layer.conv2[0].groups == layer.planes
    if torch.__version__ == 'parrots':
        if torch.cuda.is_available():
            resnet3d_csn_ir = resnet3d_csn_ir.cuda()
            imgs_gpu = imgs.cuda()
            feat = resnet3d_csn_ir(imgs_gpu)
            assert feat.shape == torch.Size([2, 2048, 1, 2, 2])
    else:
        feat = resnet3d_csn_ir(imgs)
        assert feat.shape == torch.Size([2, 2048, 1, 2, 2])

    # Set training status = False
    resnet3d_csn_ip = ResNet3dCSN(152, None, bottleneck_mode='ip')
    resnet3d_csn_ip.init_weights()
    resnet3d_csn_ip.train(False)
    for module in resnet3d_csn_ip.children():
        assert module.training is False


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


def test_c3d_backbone():
    """Test c3d backbone."""
    input_shape = (1, 3, 16, 112, 112)
    imgs = generate_backbone_demo_inputs(input_shape)

    # c3d inference test
    c3d = C3D()
    c3d.init_weights()
    c3d.train()
    feat = c3d(imgs)
    assert feat.shape == torch.Size([1, 4096])

    # c3d with bn inference test
    c3d_bn = C3D(norm_cfg=dict(type='BN3d'))
    c3d_bn.init_weights()
    c3d_bn.train()
    feat = c3d_bn(imgs)
    assert feat.shape == torch.Size([1, 4096])


def test_resnet_audio_backbone():
    """Test ResNetAudio backbone."""
    input_shape = (1, 1, 16, 16)
    spec = generate_backbone_demo_inputs(input_shape)
    # inference
    audioonly = ResNetAudio(50, None)
    audioonly.init_weights()
    audioonly.train()
    feat = audioonly(spec)
    assert feat.shape == torch.Size([1, 1024, 2, 2])


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
