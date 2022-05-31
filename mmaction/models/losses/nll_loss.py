# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn.functional as F

from mmaction.registry import MODELS
from .base import BaseWeightedLoss


@MODELS.register_module()
class NLLLoss(BaseWeightedLoss):
    """NLL Loss.

    It will calculate NLL loss given cls_score and label.
    """

    def _forward(self, cls_score, label, **kwargs):
        """Forward function.

        Args:
            cls_score (torch.Tensor): The class score.
            label (torch.Tensor): The ground truth label.
            kwargs: Any keyword argument to be used to calculate nll loss.

        Returns:
            torch.Tensor: The returned nll loss.
        """
        loss_cls = F.nll_loss(cls_score, label, **kwargs)
        return loss_cls
