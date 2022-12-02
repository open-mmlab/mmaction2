# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Union

import torch
import torch.nn as nn

from mmaction.registry import MODELS
from .base import BaseHead


@MODELS.register_module()
class GCNHead(BaseHead):
    """The classification head for GCN.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Defaults to ``dict(type='CrossEntropyLoss')``.
        dropout (float): Probability of dropout layer. Defaults to 0.
        init_cfg (dict or list[dict]): Config to control the initialization.
            Defaults to ``dict(type='Normal', layer='Linear', std=0.01)``.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 loss_cls: Dict = dict(type='CrossEntropyLoss'),
                 dropout: float = 0.,
                 average_clips: str = 'prob',
                 init_cfg: Union[Dict, List[Dict]] = dict(
                     type='Normal', layer='Linear', std=0.01),
                 **kwargs) -> None:
        super().__init__(
            num_classes,
            in_channels,
            loss_cls=loss_cls,
            average_clips=average_clips,
            init_cfg=init_cfg,
            **kwargs)
        self.dropout_ratio = dropout
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.in_channels, self.num_classes)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward features from the upstream network.

        Args:
            x (torch.Tensor): Features from the upstream network.

        Returns:
            torch.Tensor: Classification scores with shape (B, num_classes).
        """

        N, M, C, T, V = x.shape
        x = x.view(N * M, C, T, V)
        x = self.pool(x)
        x = x.view(N, M, C)
        x = x.mean(dim=1)
        assert x.shape[1] == self.in_channels

        if self.dropout is not None:
            x = self.dropout(x)

        cls_scores = self.fc(x)
        return cls_scores
