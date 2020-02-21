import pytest
import torch
import torch.nn as nn

from mmaction.models.common import (ConvModule, build_conv_layer,
                                    build_norm_layer)


def test_build_conv_layer():
    with pytest.raises(AssertionError):
        # `type` must be in cfg
        cfg = dict(kernel='3x3')
        build_conv_layer(cfg)

    with pytest.raises(AssertionError):
        # cfg must be a dict
        cfg = ['3x3']
        build_conv_layer(cfg)

    with pytest.raises(KeyError):
        cfg = dict(type='Norm')
        build_conv_layer(cfg)

    args = dict(in_channels=3, out_channels=8, kernel_size=2)
    layer = build_conv_layer(None, **args)
    assert type(layer) == nn.Conv2d
    assert layer.in_channels == args['in_channels']
    assert layer.out_channels == args['out_channels']
    assert layer.kernel_size == (2, 2)

    cfg = dict(type='Conv')
    layer = build_conv_layer(cfg, **args)
    assert type(layer) == nn.Conv2d
    assert layer.in_channels == args['in_channels']
    assert layer.out_channels == args['out_channels']
    assert layer.kernel_size == (2, 2)


def test_conv_module():
    with pytest.raises(AssertionError):
        # conv_cfg must be a dict or None
        conv_cfg = ['conv']
        ConvModule(3, 8, 2, conv_cfg=conv_cfg)

    with pytest.raises(AssertionError):
        # norm_cfg must be a dict or None
        norm_cfg = ['norm']
        ConvModule(3, 8, 2, norm_cfg=norm_cfg)

    with pytest.raises(AssertionError):
        # order elements must be ('conv', 'norm', 'act')
        order = ['conv', 'norm', 'act']
        ConvModule(3, 8, 2, order=order)

    with pytest.raises(AssertionError):
        # order elements must be ('conv', 'norm', 'act')
        order = ('conv', 'norm')
        ConvModule(3, 8, 2, order=order)

    with pytest.raises(ValueError):
        activation = 'softmax'
        ConvModule(3, 8, 2, activation=activation)

    self = ConvModule(3, 8, 2)
    self.init_weights()
    x = torch.rand(1, 3, 256, 256)
    output = self.forward(x)
    assert output.shape == torch.Size([1, 8, 255, 255])


def test_norm_layer():
    with pytest.raises(AssertionError):
        # `type` must be in cfg
        cfg = dict()
        build_norm_layer(cfg, 3)

    with pytest.raises(AssertionError):
        # cfg must be a dict
        cfg = ['BN']
        build_norm_layer(cfg, 3)

    with pytest.raises(KeyError):
        # cfg type must be in ['BN', 'BN3d', 'SyncBN', 'GN']
        cfg = dict(type='Conv')
        build_norm_layer(cfg, 3)

    with pytest.raises(NotImplementedError):
        from mmaction.models.common import norm
        norm.norm_cfg['testBN'] = ('testBN', None)
        cfg = dict(type='testBN')
        build_norm_layer(cfg, 3)

    with pytest.raises(AssertionError):
        # profix must be int or str
        cfg = dict(type='BN')
        build_norm_layer(cfg, 3, [1, 2])

    with pytest.raises(AssertionError):
        # 'num_groups' must be in cfg when using 'GN'
        cfg = dict(type='GN')
        build_norm_layer(cfg, 3)

    cfg = dict(type='BN')
    name, layer = build_norm_layer(cfg, 3, postfix=1)
    assert type(layer) == nn.BatchNorm2d
    assert name == 'bn1'
    assert layer.num_features == 3

    cfg = dict(type='BN3d')
    name, layer = build_norm_layer(cfg, 3, postfix='2')
    assert type(layer) == nn.BatchNorm3d
    assert name == 'bn2'
    assert layer.num_features == 3

    cfg = dict(type='SyncBN')
    name, layer = build_norm_layer(cfg, 3, postfix=3)
    assert type(layer) == nn.SyncBatchNorm
    assert name == 'bn3'
    assert layer.num_features == 3

    cfg = dict(type='GN', num_groups=3)
    name, layer = build_norm_layer(cfg, 3, postfix='4')
    assert type(layer) == nn.GroupNorm
    assert layer.num_channels == 3
    assert name == 'gn4'
    assert layer.num_groups == 3
