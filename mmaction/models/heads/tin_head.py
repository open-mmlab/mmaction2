import torch.nn as nn
from mmcv.cnn.weight_init import normal_init
from mmcv.runner import load_checkpoint

from ...utils import get_root_logger
from ..registry import HEADS
from .base import AvgConsensus, BaseHead


@HEADS.register_module
class TINHead(BaseHead):
    """Class head for TIN.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        finetune (str | None): Name of finetune model. Default: None.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        consensus (dict): Consensus config dict.
            Default: dict(type='AvgConsensus', dim=1).
        dropout_ratio (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.001.
        is_shift (bool): Indicating whether the feature is shifted.
            Default: True.
        temporal_pool (bool): Indicating whether feature is temporal pooled.
            Default: False.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 finetune=None,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 spatial_type='avg',
                 consensus=dict(type='AvgConsensus', dim=1),
                 dropout_ratio=0.5,
                 init_std=0.001,
                 is_shift=True,
                 temporal_pool=False):
        super(TINHead, self).__init__(num_classes, in_channels, loss_cls)

        self.finetune = finetune
        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        self.is_shift = is_shift
        self.temporal_pool = temporal_pool

        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None

        consensus_ = consensus.copy()

        consensus_type = consensus_.pop('type')
        if consensus_type == 'AvgConsensus':
            self.consensus = AvgConsensus(**consensus_)
        else:
            self.consensus = None

        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)

        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool2d` to adaptively match the in_channels.
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        else:
            self.avg_pool = None

    def init_weights(self):
        if self.finetune is None:
            normal_init(self.fc_cls, std=self.init_std)
        elif isinstance(self.finetune, str):
            logger = get_root_logger()
            load_checkpoint(self, self.finetune, strict=False, logger=logger)
        else:
            raise TypeError('finetune must be a str or None.')

    def forward(self, x, num_segments):
        # [N * num_segs, in_channels, 7, 7]
        x = self.avg_pool(x)
        # [N * num_segs, in_channels, 1, 1]
        x = x.view(x.size(0), -1)
        # [N * num_segs, in_channels]
        if self.dropout is not None:
            x = self.dropout(x)
            # [N * num_segs, in_channels]
        cls_score = self.fc_cls(x)
        # [N * num_segs, num_classes]
        if self.is_shift and self.temporal_pool:
            cls_score = cls_score.view((-1, num_segments // 2) +
                                       cls_score.size()[1:])
            # [N * 2, num_segs // 2, num_classes]
        else:
            cls_score = cls_score.view((-1, num_segments) +
                                       cls_score.size()[1:])
            # [N, num_segs, num_classes]
        cls_score = self.consensus(cls_score)
        # [N, 1, num_classes]
        return cls_score.squeeze(1)
        # [N, num_classes]
