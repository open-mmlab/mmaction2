# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmengine.model.weight_init import normal_init
from torch import Tensor

from mmaction.registry import MODELS
from mmaction.utils import ConfigType
from .base import BaseHead


@MODELS.register_module()
class STGCNHead(BaseHead):
    """The classification head for STGCN.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict or ConfigDict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        num_person (int): Number of person. Default: 2.
        init_std (float): Std value for Initiation. Default: 0.01.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 loss_cls: ConfigType = dict(type='CrossEntropyLoss'),
                 spatial_type: str = 'avg',
                 num_person: int = 2,
                 init_std: float = 0.01,
                 **kwargs) -> None:
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)

        self.spatial_type = spatial_type
        self.num_person = num_person
        self.init_std = init_std

        self.pool = None
        if self.spatial_type == 'avg':
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        elif self.spatial_type == 'max':
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
        else:
            raise NotImplementedError

        self.fc = nn.Conv2d(self.in_channels, self.num_classes, kernel_size=1)

    def init_weights(self) -> None:
        """Initialize the model network weights."""
        normal_init(self.fc, std=self.init_std)

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        """Forward features from the upstream network.

        Args:
            x (Tensor): Features from the upstream network.

        Returns:
            Tensor: Classification scores with shape(k, num_classes).
        """

        # global pooling
        assert self.pool is not None, 'pool must be implemented.'
        x = self.pool(x)
        x = x.view(x.shape[0] // self.num_person, self.num_person, -1, 1,
                   1).mean(dim=1)

        # prediction
        x = self.fc(x)
        x = x.view(x.shape[0], -1)

        return x
