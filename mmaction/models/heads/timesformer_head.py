import torch.nn as nn
from mmcv.cnn import normal_init

from ..registry import HEADS
from .base import BaseHead


@HEADS.register_module()
class TimeSformerHead(BaseHead):
    supported_attention_type = [
        'divided_space_time', 'space_only', 'joint_space_time'
    ]

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 init_std=0.01,
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)
        self.init_std = init_std
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x):
        # [N, in_channels]
        cls_score = self.fc_cls(x)
        # [N, num_classes]
        return cls_score
