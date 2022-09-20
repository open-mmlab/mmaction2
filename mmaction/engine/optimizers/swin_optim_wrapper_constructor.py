# Copyright (c) OpenMMLab. All rights reserved.
from functools import reduce
from operator import mul
from typing import List

import torch.nn as nn
from mmengine.logging import print_log
from mmengine.optim import DefaultOptimWrapperConstructor

from mmaction.registry import OPTIM_WRAPPER_CONSTRUCTORS


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class SwinOptimWrapperConstructor(DefaultOptimWrapperConstructor):

    def add_params(self,
                   params: List[dict],
                   module: nn.Module,
                   prefix: str = '',
                   **kwargs) -> None:
        custom_keys = self.paramwise_cfg.get('custom_keys', {})

        for name, param in module.named_parameters(recurse=False):
            param_group = {'params': [param]}
            if not param.requires_grad:
                params.append(param_group)
                continue

            param_group['lr'] = self.base_lr
            if self.base_wd is not None:
                param_group['weight_decay'] = self.base_wd

            processing_keys = [
                key for key in custom_keys if key in f'{prefix}.{name}'
            ]
            if processing_keys:
                param_group['lr'] *= \
                    reduce(mul, [custom_keys[key].get('lr_mult', 1.)
                                 for key in processing_keys])
                if self.base_wd is not None:
                    param_group['weight_decay'] *= \
                        reduce(mul, [custom_keys[key].get('decay_mult', 1.)
                                     for key in processing_keys])

            params.append(param_group)

            for key, value in param_group.items():
                if key == 'params':
                    continue
                full_name = f'{prefix}.{name}' if prefix else name
                print_log(
                    f'paramwise_options -- \
                    {full_name}: {key} = {round(value, 8)}',
                    logger='current')

        for child_name, child_mod in module.named_children():
            child_prefix = f'{prefix}.{child_name}' if prefix else child_name
            self.add_params(params, child_mod, prefix=child_prefix)
