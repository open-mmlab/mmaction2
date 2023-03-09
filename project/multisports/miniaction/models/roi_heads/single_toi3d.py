# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from mmdet.registry import MODELS as MMDET_MODELS
    mmdet_imported = True
except (ImportError, ModuleNotFoundError):
    mmdet_imported = False


class TOIExtractor3D(nn.Module):
    """Extract TOI-align features from a single level feature map. A pytorch
    implement of: `Spatio-Temporal Action Detection Under Large Motion`

    <https://arxiv.org/abs/2209.02250>`_
    """

    pass
