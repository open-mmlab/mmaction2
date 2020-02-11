import mmcv
import torch.nn as nn
from mmcv.cnn.weight_init import normal_init
from torch.nn.modules.utils import _pair

from ..registry import HEADS
from .base import BaseHead


class AvgConsensus(nn.Module):
    """Average consensus module.

    Attributes:
        dim (int): Decide which dim consensus function to apply.
            Default: 1.
    """

    def __init__(self, dim=1):
        super(AvgConsensus, self).__init__()
        self.dim = dim

    def forward(self, input):
        return input.mean(dim=self.dim, keepdim=True)


@HEADS.register_module
class TSNHead(BaseHead):
    """Class head for TSN.

    Attributes:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature. Default: 1024.
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        spatial_size (int): Kernel size in pooling layer. Default: 7.
        consensus (dict): Consensus config dict.
        dropout_ratio (float): Probability of dropout layer. Default: 0.8.
        init_std (float): Std value for Initiation. Default: 0.001.
    """

    def __init__(self,
                 num_classes,
                 in_channels=2048,
                 spatial_type='avg',
                 spatial_size=7,
                 consensus=dict(type='AvgConsensus', dim=1),
                 dropout_ratio=0.4,
                 init_std=0.01):
        super(TSNHead, self).__init__(num_classes, in_channels)
        self.spatial_size = _pair(spatial_size)
        assert mmcv.is_tuple_of(self.spatial_size, int)

        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std

        consensus_type = consensus.pop('type')
        if consensus_type == 'AvgConsensus':
            self.consensus = AvgConsensus(**consensus)
        else:
            self.consensus = None

        if self.spatial_type == 'avg':
            self.avg_pool2d = nn.AvgPool2d(
                self.spatial_size, stride=1, padding=0)
        else:
            self.avg_pool2d = None

        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)

    def init_weights(self):
        normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x, num_segs):
        # [N * num_segs, in_channels, 7, 7]
        x = self.avg_pool2d(x)
        # [N * num_segs, in_channels, 1, 1]
        x = x.reshape((-1, num_segs) + x.shape[1:])
        # [N, num_segs, in_channels, 1, 1]
        x = self.consensus(x)
        # [N, 1, in_channels, 1, 1]
        x = x.squeeze(1)
        # [N, in_channels, 1, 1]
        if self.dropout is not None:
            x = self.dropout(x)
            # [N, in_channels, 1, 1]
        x = x.view(x.size(0), -1)
        # [N, in_channels]
        cls_score = self.fc_cls(x)
        # [N, num_classes]
        return cls_score
