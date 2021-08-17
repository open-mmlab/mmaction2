# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner import OPTIMIZERS
from torch.optim import SGD


@OPTIMIZERS.register_module()
class CopyOfSGD(SGD):
    """A clone of torch.optim.SGD.

    A customized optimizer could be defined like CopyOfSGD. You may derive from
    built-in optimizers in torch.optim, or directly implement a new optimizer.
    """
