# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch.nn as nn
from mmengine.model import BaseInit, update_init_info

from mmaction.registry import WEIGHT_INITIALIZERS


def conv_branch_init(conv: nn.Module, branches: int) -> None:
    """Perform initialization for a conv branch.

    Args:
        conv (nn.Module): The conv module of a branch.
        branches (int): The number of branches.
    """

    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


@WEIGHT_INITIALIZERS.register_module('ConvBranch')
class ConvBranchInit(BaseInit):
    """Initialize the module parameters of different branches.

    Args:
        name (str): The name of the target module.
    """

    def __init__(self, name: str, **kwargs) -> None:
        super(ConvBranchInit, self).__init__(**kwargs)
        self.name = name

    def __call__(self, module) -> None:
        assert hasattr(module, self.name)

        # Take a short cut to get the target module
        module = getattr(module, self.name)
        num_subset = len(module)
        for conv in module:
            conv_branch_init(conv, num_subset)

        if hasattr(module, '_params_init_info'):
            update_init_info(module, init_info=self._get_init_info())

    def _get_init_info(self) -> str:
        info = f'{self.__class__.__name__}'
        return info
