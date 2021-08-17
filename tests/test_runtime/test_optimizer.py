# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.runner import build_optimizer_constructor


class SubModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 2, kernel_size=1, groups=2)
        self.gn = nn.GroupNorm(2, 2)
        self.fc = nn.Linear(2, 2)
        self.param1 = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return x


class ExampleModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.param1 = nn.Parameter(torch.ones(1))
        self.conv1 = nn.Conv2d(3, 4, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(4, 2, kernel_size=1)
        self.bn = nn.BatchNorm2d(2)
        self.sub = SubModel()
        self.fc = nn.Linear(2, 1)

    def forward(self, x):
        return x


class PseudoDataParallel(nn.Module):

    def __init__(self):
        super().__init__()
        self.module = ExampleModel()

    def forward(self, x):
        return x


base_lr = 0.01
base_wd = 0.0001
momentum = 0.9


def check_optimizer(optimizer,
                    model,
                    prefix='',
                    bias_lr_mult=1,
                    bias_decay_mult=1,
                    norm_decay_mult=1,
                    dwconv_decay_mult=1):
    param_groups = optimizer.param_groups
    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.defaults['lr'] == base_lr
    assert optimizer.defaults['momentum'] == momentum
    assert optimizer.defaults['weight_decay'] == base_wd
    model_parameters = list(model.parameters())
    assert len(param_groups) == len(model_parameters)
    for i, param in enumerate(model_parameters):
        param_group = param_groups[i]
        assert torch.equal(param_group['params'][0], param)
        assert param_group['momentum'] == momentum
    # param1
    param1 = param_groups[0]
    assert param1['lr'] == base_lr
    assert param1['weight_decay'] == base_wd
    # conv1.weight
    conv1_weight = param_groups[1]
    assert conv1_weight['lr'] == base_lr
    assert conv1_weight['weight_decay'] == base_wd
    # conv2.weight
    conv2_weight = param_groups[2]
    assert conv2_weight['lr'] == base_lr
    assert conv2_weight['weight_decay'] == base_wd
    # conv2.bias
    conv2_bias = param_groups[3]
    assert conv2_bias['lr'] == base_lr * bias_lr_mult
    assert conv2_bias['weight_decay'] == base_wd * bias_decay_mult
    # bn.weight
    bn_weight = param_groups[4]
    assert bn_weight['lr'] == base_lr
    assert bn_weight['weight_decay'] == base_wd * norm_decay_mult
    # bn.bias
    bn_bias = param_groups[5]
    assert bn_bias['lr'] == base_lr
    assert bn_bias['weight_decay'] == base_wd * norm_decay_mult
    # sub.param1
    sub_param1 = param_groups[6]
    assert sub_param1['lr'] == base_lr
    assert sub_param1['weight_decay'] == base_wd
    # sub.conv1.weight
    sub_conv1_weight = param_groups[7]
    assert sub_conv1_weight['lr'] == base_lr
    assert sub_conv1_weight['weight_decay'] == base_wd * dwconv_decay_mult
    # sub.conv1.bias
    sub_conv1_bias = param_groups[8]
    assert sub_conv1_bias['lr'] == base_lr * bias_lr_mult
    assert sub_conv1_bias['weight_decay'] == base_wd * dwconv_decay_mult
    # sub.gn.weight
    sub_gn_weight = param_groups[9]
    assert sub_gn_weight['lr'] == base_lr
    assert sub_gn_weight['weight_decay'] == base_wd * norm_decay_mult
    # sub.gn.bias
    sub_gn_bias = param_groups[10]
    assert sub_gn_bias['lr'] == base_lr
    assert sub_gn_bias['weight_decay'] == base_wd * norm_decay_mult
    # sub.fc1.weight
    sub_fc_weight = param_groups[11]
    assert sub_fc_weight['lr'] == base_lr
    assert sub_fc_weight['weight_decay'] == base_wd
    # sub.fc1.bias
    sub_fc_bias = param_groups[12]
    assert sub_fc_bias['lr'] == base_lr * bias_lr_mult
    assert sub_fc_bias['weight_decay'] == base_wd * bias_decay_mult
    # fc1.weight
    fc_weight = param_groups[13]
    assert fc_weight['lr'] == base_lr
    assert fc_weight['weight_decay'] == base_wd
    # fc1.bias
    fc_bias = param_groups[14]
    assert fc_bias['lr'] == base_lr * bias_lr_mult
    assert fc_bias['weight_decay'] == base_wd * bias_decay_mult


def check_tsm_optimizer(optimizer, model, fc_lr5=True):
    param_groups = optimizer.param_groups
    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.defaults['lr'] == base_lr
    assert optimizer.defaults['momentum'] == momentum
    assert optimizer.defaults['weight_decay'] == base_wd
    model_parameters = list(model.parameters())
    # first_conv_weight
    first_conv_weight = param_groups[0]
    assert torch.equal(first_conv_weight['params'][0], model_parameters[1])
    assert first_conv_weight['lr'] == base_lr
    assert first_conv_weight['weight_decay'] == base_wd
    # first_conv_bias
    first_conv_bias = param_groups[1]
    assert first_conv_bias['params'] == []
    assert first_conv_bias['lr'] == base_lr * 2
    assert first_conv_bias['weight_decay'] == 0
    # normal_weight
    normal_weight = param_groups[2]
    assert torch.equal(normal_weight['params'][0], model_parameters[2])
    assert torch.equal(normal_weight['params'][1], model_parameters[7])
    assert normal_weight['lr'] == base_lr
    assert normal_weight['weight_decay'] == base_wd
    # normal_bias
    normal_bias = param_groups[3]
    assert torch.equal(normal_bias['params'][0], model_parameters[3])
    assert torch.equal(normal_bias['params'][1], model_parameters[8])
    assert normal_bias['lr'] == base_lr * 2
    assert normal_bias['weight_decay'] == 0
    # bn
    bn = param_groups[4]
    assert torch.equal(bn['params'][0], model_parameters[4])
    assert torch.equal(bn['params'][1], model_parameters[5])
    assert torch.equal(bn['params'][2], model_parameters[9])
    assert torch.equal(bn['params'][3], model_parameters[10])
    assert bn['lr'] == base_lr
    assert bn['weight_decay'] == 0
    # normal linear weight
    assert torch.equal(normal_weight['params'][2], model_parameters[11])
    # normal linear bias
    assert torch.equal(normal_bias['params'][2], model_parameters[12])
    # fc_lr5
    lr5_weight = param_groups[5]
    lr10_bias = param_groups[6]
    assert lr5_weight['lr'] == base_lr * 5
    assert lr5_weight['weight_decay'] == base_wd
    assert lr10_bias['lr'] == base_lr * 10
    assert lr10_bias['weight_decay'] == 0
    if fc_lr5:
        # lr5_weight
        assert torch.equal(lr5_weight['params'][0], model_parameters[13])
        # lr10_bias
        assert torch.equal(lr10_bias['params'][0], model_parameters[14])
    else:
        # lr5_weight
        assert lr5_weight['params'] == []
        # lr10_bias
        assert lr10_bias['params'] == []
        assert torch.equal(normal_weight['params'][3], model_parameters[13])
        assert torch.equal(normal_bias['params'][3], model_parameters[14])


def test_tsm_optimizer_constructor():
    model = ExampleModel()
    optimizer_cfg = dict(
        type='SGD', lr=base_lr, weight_decay=base_wd, momentum=momentum)
    # fc_lr5 is True
    paramwise_cfg = dict(fc_lr5=True)
    optim_constructor_cfg = dict(
        type='TSMOptimizerConstructor',
        optimizer_cfg=optimizer_cfg,
        paramwise_cfg=paramwise_cfg)
    optim_constructor = build_optimizer_constructor(optim_constructor_cfg)
    optimizer = optim_constructor(model)
    check_tsm_optimizer(optimizer, model, **paramwise_cfg)

    # fc_lr5 is False
    paramwise_cfg = dict(fc_lr5=False)
    optim_constructor_cfg = dict(
        type='TSMOptimizerConstructor',
        optimizer_cfg=optimizer_cfg,
        paramwise_cfg=paramwise_cfg)
    optim_constructor = build_optimizer_constructor(optim_constructor_cfg)
    optimizer = optim_constructor(model)
    check_tsm_optimizer(optimizer, model, **paramwise_cfg)
