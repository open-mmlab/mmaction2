import torch.nn as nn
from mmcv.cnn.weight_init import normal_init
from torch.nn.modules.utils import _pair

from ..registry import HEADS
from .base import BaseHead


@HEADS.register_module
class I3DClsHead(BaseHead):
    """Classification head for I3D.

    Attributes:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature. Default: 2048.
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        spatial_size (int | tuple[int]): Kernel size in pooling layer.
            Default: 7.
        temporal_size (int): Temporal stride in the `nn.AvgPool3d` layer.
            Default: 4.
        dropout_ratio (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
    """

    def __init__(self,
                 num_classes,
                 in_channels=2048,
                 spatial_type='avg',
                 spatial_size=7,
                 temporal_size=4,
                 dropout_ratio=0.5,
                 init_std=0.01):
        super(I3DClsHead, self).__init__(num_classes, in_channels)
        if not isinstance(spatial_size, int):
            self.spatial_size = spatial_size
        else:
            self.spatial_size = _pair(spatial_size)
        self.spatial_type = spatial_type
        self.temporal_size = temporal_size
        self.pool_size = (self.temporal_size, ) + self.spatial_size
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)

        if self.spatial_type == 'avg':
            self.avg_pool = nn.AvgPool3d(self.pool_size, stride=1, padding=0)
        else:
            self.avg_pool = None

    def init_weights(self):
        normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x):
        x = self.avg_pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.shape[0], -1)
        cls_score = self.fc_cls(x)
        return cls_score
