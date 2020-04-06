import torch
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.conv import _ConvNd

from .default_constructor import DefaultOptimizerConstructor
from .registry import OPTIMIZER_BUILDERS


@OPTIMIZER_BUILDERS.register_module
class TSMOptimizerConstructor(DefaultOptimizerConstructor):
    """Optimizer constructor for TSM model.

    This constructor build optimizer in different ways from the default one.
    1. Parameters of the first conv layer have default lr and weight decay.
    2. Parameters of BN layers have default lr and zero weight decay.
    3. If the field "fc_lr5" in paramwise_cfg is set to True, then parameters
       of fc layers have a 5x lr multiplier and 10x weight decay multiplier.
    4. Weights of other layers have default lr and weight decay, and biases
       have a 2x lr multiplier and zero weight decay.
    """

    def add_params(self, params, model):
        fc_lr5 = self.paramwise_cfg['fc_lr5']
        first_conv_weight = []
        normal_weight = []
        normal_bias = []
        lr5_weight = []
        lr10_bias = []
        bn = []
        conv_cnt = 0

        for m in model.modules():
            if isinstance(m, _ConvNd):
                m_params = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(m_params[0])
                else:
                    normal_weight.append(m_params[0])
            elif isinstance(m, torch.nn.Linear):
                m_params = list(m.parameters())
                if fc_lr5:
                    lr5_weight.append(m_params[0])
                    lr10_bias.append(m_params[1])
                else:
                    normal_weight.append(m_params[0])
                    normal_bias.append(m_params[1])
            elif isinstance(m, _BatchNorm):
                for param in list(m.parameters()):
                    if param.requires_grad:
                        bn.append(param)
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError(
                        'New atomic module type: {}. '
                        'Need to give it a learning policy'.format(type(m)))
        params.append({
            'params': first_conv_weight,
            'lr': self.base_lr,
            'weight_decay': self.base_wd,
        })
        params.append({
            'params': normal_weight,
            'lr': self.base_lr,
            'weight_decay': self.base_wd,
        })
        params.append({
            'params': normal_bias,
            'lr': self.base_lr * 2,
            'weight_decay': 0,
        })
        params.append({
            'params': bn,
            'lr': self.base_lr,
            'weight_decay': 0,
        })
        params.append({
            'params': lr5_weight,
            'lr': self.base_lr * 5,
            'weight_decay': self.base_wd,
        })
        params.append({
            'params': lr10_bias,
            'lr': self.base_lr * 10,
            'weight_decay': 0,
        })
