import pytest
import torch
import torch.nn as nn

from mmaction.models.common import (ConvModule, build_activation_layer,
                                    build_conv_layer, build_norm_layer)


def test_build_conv_layer():
    with pytest.raises(TypeError):
        # `type` must be in cfg
        cfg = dict(kernel='3x3')
        build_conv_layer(cfg)

    with pytest.raises(TypeError):
        # cfg must be a dict
        cfg = ['3x3']
        build_conv_layer(cfg)

    with pytest.raises(KeyError):
        # cfg `type` must be in ['Conv', 'Conv3d']
        cfg = dict(type='Norm')
        build_conv_layer(cfg)

    # build layer with cfg=None
    args = dict(in_channels=3, out_channels=8, kernel_size=2)
    layer = build_conv_layer(None, **args)
    assert type(layer) == nn.Conv2d
    assert layer.in_channels == args['in_channels']
    assert layer.out_channels == args['out_channels']
    assert layer.kernel_size == (2, 2)

    # build layer indicating cfg
    cfg = dict(type='Conv')
    layer = build_conv_layer(cfg, **args)
    assert type(layer) == nn.Conv2d
    assert layer.in_channels == args['in_channels']
    assert layer.out_channels == args['out_channels']
    assert layer.kernel_size == (2, 2)


def test_build_activation_layer():
    with pytest.raises(TypeError):
        # `type` must be in cfg
        cfg = dict()
        build_activation_layer(cfg)

    with pytest.raises(TypeError):
        # cfg must be a dict
        cfg = ['ReLU']
        build_activation_layer(cfg)

    with pytest.raises(KeyError):
        # cfg `type` must be in
        # ['ReLU', 'LeakyReLU', 'PReLU', 'RReLU', 'ReLU6', 'SELU', 'CELU']
        cfg = dict(type='relu')
        build_activation_layer(cfg)

    with pytest.raises(NotImplementedError):
        # None in activation_cfg values.
        from mmaction.models.common import activation
        activation_cfg = activation.activation_cfg
        activation_cfg['testReLU'] = None
        cfg = dict(type='testReLU')
        build_activation_layer(cfg)

    from mmaction.models.common import activation
    activation_cfg = activation.activation_cfg
    activation_cfg.pop('testReLU', None)

    # test each type of activation layer in activation_cfg
    cfg = dict()
    for activation_type in activation_cfg:
        cfg['type'] = activation_type
        layer = build_activation_layer(cfg)
        assert type(layer) == activation_cfg[activation_type]


def test_build_norm_layer():
    with pytest.raises(TypeError):
        # `type` must be in cfg
        cfg = dict()
        build_norm_layer(cfg, 3)

    with pytest.raises(TypeError):
        # cfg must be a dict
        cfg = ['BN']
        build_norm_layer(cfg, 3)

    with pytest.raises(KeyError):
        # cfg `type` must be in ['BN', 'BN3d', 'SyncBN', 'GN']
        cfg = dict(type='Conv')
        build_norm_layer(cfg, 3)

    with pytest.raises(NotImplementedError):
        # None in norm_cfg values.
        from mmaction.models.common import norm
        norm_cfg = norm.norm_cfg
        norm_cfg['testBN'] = ('testBN', None)
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

    from mmaction.models.common import norm
    norm_cfg = norm.norm_cfg
    norm_cfg.pop('testBN', None)

    # test each type of norm layer in norm_cfg
    cfg = dict()
    postfix = '_test'
    for norm_type in norm_cfg:
        cfg['type'] = norm_type
        if norm_type == 'GN':
            cfg['num_groups'] = 3
        name, layer = build_norm_layer(cfg, 3, postfix=postfix)
        assert name == norm_cfg[norm_type][0] + postfix
        assert type(layer) == norm_cfg[norm_type][1]
        if norm_type != 'GN':
            assert layer.num_features == 3
        else:
            assert layer.num_channels == 3
            assert layer.num_groups == cfg['num_groups']


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

    # ConvModule with no norm and act
    layer = ConvModule(3, 8, 2)
    layer.init_weights()
    x = torch.rand(1, 3, 256, 256)
    output = layer.forward(x)
    assert output.shape == torch.Size([1, 8, 255, 255])

    # ConvModule with norm and act
    norm_cfg = dict(type='BN')
    act_cfg = dict(type='LeakyReLU')
    layer = ConvModule(3, 8, 2, norm_cfg=norm_cfg, act_cfg=act_cfg)
    layer.init_weights()
    x = torch.rand(1, 3, 256, 256)
    output = layer.forward(x)
    assert output.shape == torch.Size([1, 8, 255, 255])

    # test Conv(2+1)d with BN3d and LeakyReLU
    norm_cfg = dict(type='BN3d')
    act_cfg = dict(type='LeakyReLU')
    conv_cfg = dict(type='Conv(2+1)d')
    order = ('norm', 'conv', 'act')
    layer = ConvModule(
        3,
        8,
        2,
        conv_cfg=conv_cfg,
        norm_cfg=norm_cfg,
        act_cfg=act_cfg,
        order=order)
    layer.init_weights()
    x = torch.rand(1, 3, 8, 256, 256)
    output = layer.forward(x)
    assert output.shape == torch.Size([1, 8, 7, 255, 255])
