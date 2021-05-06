import torch.nn as nn

# from mmcv.cnn import normal_init
from mmaction.utils import trunc_normal_
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
                 init_std=0.02,
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)
        self.init_std = init_std
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)

    def init_weights(self):
        """Initiate the parameters from scratch."""
        trunc_normal_(self.fc_cls.weight, std=self.init_std)
        nn.init.constant_(self.fc_cls.bias, 0)
        # normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x):
        # [N, in_channels]
        cls_score = self.fc_cls(x)
        # [N, num_classes]
        return cls_score
