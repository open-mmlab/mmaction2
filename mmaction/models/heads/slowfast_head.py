# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch
from mmengine.model.weight_init import normal_init
from torch import Tensor, nn

from mmaction.registry import MODELS
from mmaction.utils import ConfigType
from .base import BaseHead


@MODELS.register_module()
class SlowFastHead(BaseHead):
    """The classification head for SlowFast.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict or ConfigDict): Config for building loss.
            Default: dict(type='CrossEntropyLoss').
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        dropout_ratio (float): Probability of dropout layer. Default: 0.8.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 loss_cls: ConfigType = dict(type='CrossEntropyLoss'),
                 spatial_type: str = 'avg',
                 dropout_ratio: float = 0.8,
                 init_std: float = 0.01,
                 **kwargs) -> None:

        super().__init__(num_classes, in_channels, loss_cls, **kwargs)
        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std

        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_cls = nn.Linear(in_channels, num_classes)

        if self.spatial_type == 'avg':
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool = None

    def init_weights(self) -> None:
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x: Tuple[Tensor], **kwargs) -> None:
        """Defines the computation performed at every call.

        Args:
            x (tuple[torch.Tensor]): The input data.

        Returns:
            Tensor: The classification scores for input samples.
        """
        # ([N, channel_slow, T1, H, W], [(N, channel_fast, T2, H, W)])
        x_slow, x_fast = x
        # ([N, channel_slow, 1, 1, 1], [N, channel_fast, 1, 1, 1])
        x_slow = self.avg_pool(x_slow)
        x_fast = self.avg_pool(x_fast)
        # [N, channel_fast + channel_slow, 1, 1, 1]
        x = torch.cat((x_fast, x_slow), dim=1)

        if self.dropout is not None:
            x = self.dropout(x)

        # [N x C]
        x = x.view(x.size(0), -1)
        # [N x num_classes]
        cls_score = self.fc_cls(x)

        return cls_score
