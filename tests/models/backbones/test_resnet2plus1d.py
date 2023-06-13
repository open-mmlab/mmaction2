# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm

from mmaction.models import ResNet2Plus1d
from mmaction.testing import generate_backbone_demo_inputs


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
