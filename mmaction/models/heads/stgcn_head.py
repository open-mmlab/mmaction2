# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import normal_init

from ..builder import HEADS
from .base import BaseHead


@HEADS.register_module()
class STGCNHead(BaseHead):
    """The classification head for STGCN.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        num_person (int): Number of person. Default: 2.
        init_std (float): Std value for Initiation. Default: 0.01.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 spatial_type='avg',
                 num_person=2,
                 init_std=0.01):
        super().__init__(num_classes, in_channels, loss_cls)

        self.spatial_type = spatial_type
        self.in_channels = in_channels
        self.num_classes = num_classes
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

    def init_weights(self):
        normal_init(self.fc, std=self.init_std)

    def forward(self, x):
        # global pooling
        assert self.pool is not None
        x = self.pool(x)
        x = x.view(x.shape[0] // self.num_person, self.num_person, -1, 1,
                   1).mean(dim=1)

        # prediction
        x = self.fc(x)
        x = x.view(x.shape[0], -1)

        return x
