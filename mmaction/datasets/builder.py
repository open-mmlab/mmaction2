# Copyright (c) OpenMMLab. All rights reserved.
import platform

from mmengine import TRANSFORMS as MMEngine_TRANSFORMS
from mmengine.registry import DATASETS as MMEngine_DATASETS
from mmengine.registry import Registry
from torch.utils.data import DataLoader

if platform.system() != 'Windows':
    # https://github.com/pytorch/pytorch/issues/973
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    hard_limit = rlimit[1]
    soft_limit = min(4096, hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

DATASETS = Registry('dataset', parent=MMEngine_DATASETS)
TRANSFORMS = Registry('transforms', parent=MMEngine_TRANSFORMS)
BLENDINGS = Registry('blending')
