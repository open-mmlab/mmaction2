import itertools

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from ..registry import HEADS
from .base import BaseHead


class RelationModule(nn.Module):

    def __init__(self, hidden_dim, num_segments, num_classes):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_segments = num_segments
        self.num_classes = num_classes
        bottleneck_dim = 512
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.num_segments * self.hidden_dim, bottleneck_dim),
            nn.ReLU(), nn.Linear(bottleneck_dim, self.num_classes))

    def forward(self, x):
        # [N, num_segs * hidden_dim]
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class RelationModuleMultiScale(nn.Module):

    def __init__(self, hidden_dim, num_segments, num_classes):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_segments = num_segments
        self.num_classes = num_classes

        # generate the multiple frame relations
        self.scales = range(num_segments, 1, -1)

        self.relations_scales = []
        self.subsample_scales = []
        max_subsample = 3
        for scale in self.scales:
            relations_scale = list(
                itertools.combinations(range(self.num_segments), scale))
            self.relations_scales.append(relations_scale)
            self.subsample_scales.append(
                min(max_subsample, len(relations_scale)))
        assert len(self.relations_scales[0]) == 1

        bottleneck_dim = 256
        self.fc_fusion_scales = nn.ModuleList()
        for scale in self.scales:
            fc_fusion = nn.Sequential(
                nn.ReLU(),
                nn.Linear(scale * self.hidden_dim, bottleneck_dim),
                nn.ReLU(),
                nn.Linear(bottleneck_dim, self.num_classes),
            )
            self.fc_fusion_scales.append(fc_fusion)

    def forward(self, x):
        # the first one is the largest scale
        act_all = x[:, self.relations_scales[0][0], :]
        act_all = act_all.view(
            act_all.size(0), self.scales[0] * self.hidden_dim)
        act_all = self.fc_fusion_scales[0](act_all)

        for scaleID in range(1, len(self.scales)):
            # iterate over the scales
            idx_relations_randomsample = np.random.choice(
                len(self.relations_scales[scaleID]),
                self.subsample_scales[scaleID],
                replace=False)
            for idx in idx_relations_randomsample:
                act_relation = x[:, self.relations_scales[scaleID][idx], :]
                act_relation = act_relation.view(
                    act_relation.size(0),
                    self.scales[scaleID] * self.hidden_dim)
                act_relation = self.fc_fusion_scales[scaleID](act_relation)
                act_all += act_relation
        return act_all


@HEADS.register_module()
class TRNHead(BaseHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 num_segments=8,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 spatial_type='avg',
                 relation_type='TRNMultiScale',
                 hidden_dim=256,
                 dropout_ratio=0.4,
                 init_std=0.001,
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)

        assert relation_type in ['TRN', 'TRNMultiScale']
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_segments = num_segments
        self.spatial_type = spatial_type
        self.relation_type = relation_type
        self.hidden_dim = hidden_dim
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std

        if self.relation_type == 'TRN':
            self.consensus = RelationModule(self.hidden_dim, self.num_segments,
                                            self.num_classes)
        elif self.relation_type == 'TRNMultiScale':
            self.consensus = RelationModuleMultiScale(self.hidden_dim,
                                                      self.num_segments,
                                                      self.num_classes)

        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_cls = nn.Linear(self.in_channels, self.hidden_dim)

        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool2d` to adaptively match the in_channels.
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        else:
            self.avg_pool = None

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x, num_segs):
        # [N * num_segs, in_channels, 7, 7]
        if self.avg_pool is not None:
            x = self.avg_pool(x)
        # [N * num_segs, in_channels, 1, 1]
        x = torch.flatten(x, 1)
        # [N * num_segs, in_channels]
        if self.dropout is not None:
            x = self.dropout(x)

        # [N, num_segs, hidden_dim]
        cls_score = self.fc_cls(x)
        cls_score = cls_score.view((-1, self.num_segments) +
                                   cls_score.size()[1:])

        # [N, num_classes]
        cls_score = self.consensus(cls_score)
        return cls_score
