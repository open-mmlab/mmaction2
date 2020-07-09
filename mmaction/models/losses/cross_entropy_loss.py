import torch.nn.functional as F

from ..registry import LOSSES
from .base import BaseWeightedLoss


@LOSSES.register_module()
class CrossEntropyLoss(BaseWeightedLoss):
    """Cross Entropy Loss."""

    def _forward(self, cls_score, label, **kwargs):
        """Forward function.

        Args:
            cls_score (torch.Tensor): The class score.
            label (torch.Tensor): The ground truth label.
            kwargs: Any keyword argument to be used to calculate
                CrossEntropy loss.

        Returns:
            torch.Tensor: The returned CrossEntropy loss.
        """
        loss_cls = F.cross_entropy(cls_score, label, **kwargs)
        return loss_cls


@LOSSES.register_module()
class BCELossWithLogits(BaseWeightedLoss):
    """Binary Cross Entropy Loss with logits."""

    def _forward(self, cls_score, label, **kwargs):
        """Forward function.

        Args:
            cls_score (torch.Tensor): The class score.
            label (torch.Tensor): The ground truth label.
            kwargs: Any keyword argument to be used to calculate
                bce loss with logits.

        Returns:
            torch.Tensor: The returned bce loss with logits.
        """
        loss_cls = F.binary_cross_entropy_with_logits(cls_score, label,
                                                      **kwargs)
        return loss_cls
