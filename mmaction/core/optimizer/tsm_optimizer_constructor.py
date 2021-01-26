import torch
from mmcv.runner import OPTIMIZER_BUILDERS, DefaultOptimizerConstructor
from mmcv.utils import SyncBatchNorm, _BatchNorm, _ConvNd


@OPTIMIZER_BUILDERS.register_module()
class TSMOptimizerConstructor(DefaultOptimizerConstructor):
    """Optimizer constructor in TSM model.

    This constructor builds optimizer in different ways from the default one.

    1. Parameters of the first conv layer have default lr and weight decay.
    2. Parameters of BN layers have default lr and zero weight decay.
    3. If the field "fc_lr5" in paramwise_cfg is set to True, the parameters
       of the last fc layer in cls_head have 5x lr multiplier and 10x weight
       decay multiplier.
    4. Weights of other layers have default lr and weight decay, and biases
       have a 2x lr multiplier and zero weight decay.
    """

    def add_params(self, params, model):
        """Add parameters and their corresponding lr and wd to the params.

        Args:
            params (list): The list to be modified, containing all parameter
                groups and their corresponding lr and wd configurations.
            model (nn.Module): The model to be trained with the optimizer.
        """
        # use fc_lr5 to determine whether to specify higher multi-factor
        # for fc layer weights and bias.
        fc_lr5 = self.paramwise_cfg['fc_lr5']
        first_conv_weight = []
        first_conv_bias = []
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
                    if len(m_params) == 2:
                        first_conv_bias.append(m_params[1])
                else:
                    normal_weight.append(m_params[0])
                    if len(m_params) == 2:
                        normal_bias.append(m_params[1])
            elif isinstance(m, torch.nn.Linear):
                m_params = list(m.parameters())
                normal_weight.append(m_params[0])
                if len(m_params) == 2:
                    normal_bias.append(m_params[1])
            elif isinstance(m,
                            (_BatchNorm, SyncBatchNorm, torch.nn.GroupNorm)):
                for param in list(m.parameters()):
                    if param.requires_grad:
                        bn.append(param)
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError(f'New atomic module type: {type(m)}. '
                                     'Need to give it a learning policy')

        # pop the cls_head fc layer params
        last_fc_weight = normal_weight.pop()
        last_fc_bias = normal_bias.pop()
        if fc_lr5:
            lr5_weight.append(last_fc_weight)
            lr10_bias.append(last_fc_bias)
        else:
            normal_weight.append(last_fc_weight)
            normal_bias.append(last_fc_bias)

        params.append({
            'params': first_conv_weight,
            'lr': self.base_lr,
            'weight_decay': self.base_wd
        })
        params.append({
            'params': first_conv_bias,
            'lr': self.base_lr * 2,
            'weight_decay': 0
        })
        params.append({
            'params': normal_weight,
            'lr': self.base_lr,
            'weight_decay': self.base_wd
        })
        params.append({
            'params': normal_bias,
            'lr': self.base_lr * 2,
            'weight_decay': 0
        })
        params.append({'params': bn, 'lr': self.base_lr, 'weight_decay': 0})
        params.append({
            'params': lr5_weight,
            'lr': self.base_lr * 5,
            'weight_decay': self.base_wd
        })
        params.append({
            'params': lr10_bias,
            'lr': self.base_lr * 10,
            'weight_decay': 0
        })
