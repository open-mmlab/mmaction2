# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from mmaction.engine.optimizers import SwinOptimWrapperConstructor


class SubModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 2, kernel_size=1, groups=2)
        self.gn = nn.GroupNorm(2, 2)
        self.fc = nn.Linear(2, 2)
        self.param1 = nn.Parameter(torch.ones(1))


class ExampleModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.param1 = nn.Parameter(torch.ones(1))
        self.conv1 = nn.Conv2d(3, 4, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(4, 2, kernel_size=1)
        self.bn = nn.BatchNorm2d(2)
        self.sub = SubModel()
        self.fc = nn.Linear(2, 1)


base_lr = 0.01
base_wd = 0.0001
betas = (0.9, 0.999)


def test_swin_optim_wrapper_constructor():
    model = ExampleModel()
    optim_wrapper_cfg = dict(
        optimizer=dict(
            type='AdamW', lr=base_lr, weight_decay=base_wd, betas=betas))
    paramwise_cfg = {
        'base.param1': dict(lr_mult=2.),
        'base.conv1.weight': dict(lr_mult=3.),
        'bn': dict(decay_mult=0.),
        'sub': dict(lr_mult=0.1),
        'sub.conv1.bias': dict(decay_mult=0.1),
        'gn': dict(decay_mult=0.),
    }
    constructor = SwinOptimWrapperConstructor(optim_wrapper_cfg, paramwise_cfg)
    optim_wrapper = constructor(model)

    optimizer = optim_wrapper.optimizer
    param_groups = optimizer.param_groups
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.defaults['lr'] == base_lr
    assert optimizer.defaults['weight_decay'] == base_wd
    model_parameters = list(model.parameters())
    assert len(param_groups) == len(model_parameters)
    for i, param in enumerate(model_parameters):
        param_group = param_groups[i]
        assert torch.equal(param_group['params'][0], param)
        assert param_group['betas'] == betas

    # param1
    param1 = param_groups[0]
    assert param1['lr'] == base_lr * paramwise_cfg['base.param1']['lr_mult']
    assert param1['weight_decay'] == base_wd
    # conv1.weight
    conv1_weight = param_groups[1]
    assert conv1_weight['lr'] == \
           base_lr * paramwise_cfg['base.conv1.weight']['lr_mult']
    assert conv1_weight['weight_decay'] == base_wd
    # conv2.weight
    conv2_weight = param_groups[2]
    assert conv2_weight['lr'] == base_lr
    assert conv2_weight['weight_decay'] == base_wd
    # conv2.bias
    conv2_bias = param_groups[3]
    assert conv2_bias['lr'] == base_lr
    assert conv2_bias['weight_decay'] == base_wd
    # bn.weight
    bn_weight = param_groups[4]
    assert bn_weight['lr'] == base_lr
    assert bn_weight['weight_decay'] == \
           base_wd * paramwise_cfg['bn']['decay_mult']
    # bn.bias
    bn_bias = param_groups[5]
    assert bn_bias['lr'] == base_lr
    assert bn_bias['weight_decay'] == \
           base_wd * paramwise_cfg['bn']['decay_mult']
    # sub.param1
    sub_param1 = param_groups[6]
    assert sub_param1['lr'] == base_lr * paramwise_cfg['sub']['lr_mult']
    assert sub_param1['weight_decay'] == base_wd
    # sub.conv1.weight
    sub_conv1_weight = param_groups[7]
    assert sub_conv1_weight['lr'] == base_lr * paramwise_cfg['sub']['lr_mult']
    assert sub_conv1_weight['weight_decay'] == base_wd
    # sub.conv1.bias
    sub_conv1_bias = param_groups[8]
    assert sub_conv1_bias['lr'] == base_lr * paramwise_cfg['sub']['lr_mult']
    assert sub_conv1_bias['weight_decay'] == \
           base_wd * paramwise_cfg['sub.conv1.bias']['decay_mult']
    # sub.gn.weight
    sub_gn_weight = param_groups[9]
    assert sub_gn_weight['lr'] == base_lr * paramwise_cfg['sub']['lr_mult']
    assert sub_gn_weight['weight_decay'] == \
           base_wd * paramwise_cfg['gn']['decay_mult']
    # sub.gn.bias
    sub_gn_bias = param_groups[10]
    assert sub_gn_bias['lr'] == base_lr * paramwise_cfg['sub']['lr_mult']
    assert sub_gn_bias['weight_decay'] == \
           base_wd * paramwise_cfg['gn']['decay_mult']
    # sub.fc.weight
    sub_fc_weight = param_groups[11]
    assert sub_fc_weight['lr'] == base_lr * paramwise_cfg['sub']['lr_mult']
    assert sub_fc_weight['weight_decay'] == base_wd
    # sub.fc.bias
    sub_fc_bias = param_groups[12]
    assert sub_fc_bias['lr'] == base_lr * paramwise_cfg['sub']['lr_mult']
    assert sub_fc_bias['weight_decay'] == base_wd
    # fc.weight
    fc_weight = param_groups[13]
    assert fc_weight['lr'] == base_lr
    assert fc_weight['weight_decay'] == base_wd
    # fc.bias
    fc_bias = param_groups[14]
    assert fc_bias['lr'] == base_lr
    assert fc_bias['weight_decay'] == base_wd
