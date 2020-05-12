import torch.nn.functional as F

from ..registry import LOSSES
from .base import BaseWeightedLoss


@LOSSES.register_module
class CrossEntropyLoss(BaseWeightedLoss):
    """Cross Entropy Loss."""

    def _forward(self, cls_score, label, **kwargs):
        loss_cls = F.cross_entropy(cls_score, label, **kwargs)
        return loss_cls


@LOSSES.register_module
class BCELossWithLogits(BaseWeightedLoss):
    """Binary Cross Entropy Loss with logits."""

    def _forward(self, cls_score, label, **kwargs):
        loss_cls = F.binary_cross_entropy_with_logits(cls_score, label,
                                                      **kwargs)
        return loss_cls
