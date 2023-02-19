# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
from mmengine.model.weight_init import normal_init

from mmaction.registry import MODELS
from .base import BaseHead


@MODELS.register_module()
class RGBPoseHead(BaseHead):
    """The classification head for Slowfast.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (tuple[int]): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Defaults to ``dict(type='CrossEntropyLoss')``.
        loss_components (list[str]): The components of the loss.
            Defaults to ``['rgb', 'pose']``.
        loss_weights (float or tuple[float]): The weights of the losses.
            Defaults to 1.
        dropout (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: Tuple[int],
                 loss_cls: Dict = dict(type='CrossEntropyLoss'),
                 loss_components: List[str] = ['rgb', 'pose'],
                 loss_weights: Union[float, Tuple[float]] = 1.,
                 dropout: float = 0.5,
                 init_std: float = 0.01,
                 **kwargs) -> None:
        super().__init__(num_classes, in_channels, loss_cls,
                         loss_components=loss_components,
                         loss_weights=loss_weights, **kwargs)
        if isinstance(dropout, float):
            dropout = {'rgb': dropout, 'pose': dropout}
        assert isinstance(dropout, dict)

        self.dropout = dropout
        self.init_std = init_std
        self.in_channels = in_channels

        self.dropout_rgb = nn.Dropout(p=self.dropout['rgb'])
        self.dropout_pose = nn.Dropout(p=self.dropout['pose'])

        self.fc_rgb = nn.Linear(in_channels[0], num_classes)
        self.fc_pose = nn.Linear(in_channels[1], num_classes)
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def init_weights(self) -> None:
        """Initiate the parameters from scratch."""
        normal_init(self.fc_rgb, std=self.init_std)
        normal_init(self.fc_pose, std=self.init_std)

    def forward(self, x: Tuple[torch.Tensor]) -> Dict:
        """Defines the computation performed at every call."""
        x_rgb, x_pose = self.avg_pool(x[0]), self.avg_pool(x[1])
        x_rgb = x_rgb.view(x_rgb.size(0), -1)
        x_pose = x_pose.view(x_pose.size(0), -1)

        x_rgb = self.dropout_rgb(x_rgb)
        x_pose = self.dropout_pose(x_pose)

        cls_scores = dict()
        cls_scores['rgb'] = self.fc_rgb(x_rgb)
        cls_scores['pose'] = self.fc_pose(x_pose)

        return cls_scores
