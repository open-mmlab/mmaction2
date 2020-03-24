import pytest
import torch
import torch.nn as nn

from mmaction.core import build_optimizer
from mmaction.core.optimizer.registry import TORCH_OPTIMIZERS


class ExampleModel(nn.Module):

    def __init__(self):
        super(ExampleModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3)
        self.bn = nn.BatchNorm2d(8)
        self.gn = nn.GroupNorm(3, 8)

    def forward(self, imgs):
        return imgs


def test_build_optimizer():
    with pytest.raises(TypeError):
        optimizer_cfg = dict(paramwise_options=['error'])
        model = ExampleModel()
        build_optimizer(model, optimizer_cfg)

    with pytest.raises(ValueError):
        optimizer_cfg = dict(
            paramwise_options=dict(bias_decay_mult=1, norm_decay_mult=1),
            lr=0.0001,
            weight_decay=None)
        model = ExampleModel()
        build_optimizer(model, optimizer_cfg)

    base_lr = 0.0001
    base_wd = 0.0002
    momentum = 0.9

    # basic config with ExampleModel
    optimizer_cfg = dict(
        type='SGD', lr=base_lr, weight_decay=base_wd, momentum=momentum)
    model = ExampleModel()
    optimizer = build_optimizer(model, optimizer_cfg)
    param_dict = dict(model.named_parameters())
    param_groups = optimizer.param_groups[0]
    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.defaults['lr'] == 0.0001
    assert optimizer.defaults['momentum'] == 0.9
    assert optimizer.defaults['weight_decay'] == 0.0002
    assert len(param_groups['params']) == 6
    assert torch.equal(param_groups['params'][0], param_dict['conv1.weight'])
    assert torch.equal(param_groups['params'][1], param_dict['conv1.bias'])
    assert torch.equal(param_groups['params'][2], param_dict['bn.weight'])
    assert torch.equal(param_groups['params'][3], param_dict['bn.bias'])
    assert torch.equal(param_groups['params'][4], param_dict['gn.weight'])
    assert torch.equal(param_groups['params'][5], param_dict['gn.bias'])

    # basic config with Parallel model
    model = torch.nn.DataParallel(ExampleModel())
    optimizer = build_optimizer(model, optimizer_cfg)
    param_dict = dict(model.named_parameters())
    param_groups = optimizer.param_groups[0]
    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.defaults['lr'] == 0.0001
    assert optimizer.defaults['momentum'] == 0.9
    assert optimizer.defaults['weight_decay'] == 0.0002
    assert len(param_groups['params']) == 6
    assert torch.equal(param_groups['params'][0],
                       param_dict['module.conv1.weight'])
    assert torch.equal(param_groups['params'][1],
                       param_dict['module.conv1.bias'])
    assert torch.equal(param_groups['params'][2],
                       param_dict['module.bn.weight'])
    assert torch.equal(param_groups['params'][3], param_dict['module.bn.bias'])
    assert torch.equal(param_groups['params'][4],
                       param_dict['module.gn.weight'])
    assert torch.equal(param_groups['params'][5], param_dict['module.gn.bias'])

    # Empty paramwise_options with ExampleModel
    optimizer_cfg['paramwise_options'] = dict()
    model = ExampleModel()
    optimizer = build_optimizer(model, optimizer_cfg)
    param_groups = optimizer.param_groups
    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.defaults['lr'] == 0.0001
    assert optimizer.defaults['momentum'] == 0.9
    assert optimizer.defaults['weight_decay'] == 0.0002
    for i, (name, param) in enumerate(model.named_parameters()):
        param_group = param_groups[i]
        assert param_group['params'] == [param]
        assert param_group['momentum'] == 0.9
        assert param_group['lr'] == 0.0001
        assert param_group['weight_decay'] == 0.0002

    # Empty paramwise_options with ExampleModel and no grad
    for param in model.parameters():
        param.requires_grad = False
    optimizer = build_optimizer(model, optimizer_cfg)
    param_groups = optimizer.param_groups
    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.defaults['lr'] == 0.0001
    assert optimizer.defaults['momentum'] == 0.9
    assert optimizer.defaults['weight_decay'] == 0.0002
    for i, (name, param) in enumerate(model.named_parameters()):
        param_group = param_groups[i]
        assert param_group['params'] == [param]
        assert param_group['momentum'] == 0.9
        assert param_group['lr'] == 0.0001
        assert param_group['weight_decay'] == 0.0002

    # paramwise_options with ExampleModel
    paramwise_options = dict(
        bias_lr_mult=0.9, bias_decay_mult=0.8, norm_decay_mult=0.7)
    optimizer_cfg['paramwise_options'] = paramwise_options
    model = ExampleModel()
    optimizer = build_optimizer(model, optimizer_cfg)
    param_groups = optimizer.param_groups
    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.defaults['lr'] == 0.0001
    assert optimizer.defaults['momentum'] == 0.9
    assert optimizer.defaults['weight_decay'] == 0.0002
    for i, (name, param) in enumerate(model.named_parameters()):
        param_group = param_groups[i]
        assert param_group['params'] == [param]
        assert param_group['momentum'] == 0.9
    assert param_groups[0]['lr'] == 0.0001
    assert param_groups[0]['weight_decay'] == 0.0002
    assert param_groups[1]['lr'] == 0.0001 * 0.9
    assert param_groups[1]['weight_decay'] == 0.0002 * 0.8
    assert param_groups[2]['lr'] == 0.0001
    assert param_groups[2]['weight_decay'] == 0.0002 * 0.7
    assert param_groups[3]['lr'] == 0.0001
    assert param_groups[3]['weight_decay'] == 0.0002 * 0.7
    assert param_groups[4]['lr'] == 0.0001
    assert param_groups[4]['weight_decay'] == 0.0002 * 0.7
    assert param_groups[5]['lr'] == 0.0001
    assert param_groups[5]['weight_decay'] == 0.0002 * 0.7

    # paramwise_options with ExampleModel and no grad
    for param in model.parameters():
        param.requires_grad = False
    optimizer = build_optimizer(model, optimizer_cfg)
    param_groups = optimizer.param_groups
    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.defaults['lr'] == 0.0001
    assert optimizer.defaults['momentum'] == 0.9
    assert optimizer.defaults['weight_decay'] == 0.0002
    for i, (name, param) in enumerate(model.named_parameters()):
        param_group = param_groups[i]
        assert param_group['params'] == [param]
        assert param_group['momentum'] == 0.9
        assert param_group['lr'] == 0.0001
        assert param_group['weight_decay'] == 0.0002

    torch_optimizers = [
        'ASGD', 'Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'LBFGS',
        'Optimizer', 'RMSprop', 'Rprop', 'SGD', 'SparseAdam'
    ]
    assert set(torch_optimizers).issubset(set(TORCH_OPTIMIZERS))
