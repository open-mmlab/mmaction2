# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F

from mmaction.registry import MODELS
from .base import BaseWeightedLoss


@MODELS.register_module()
class MeanSquareErrorLoss(BaseWeightedLoss):
    """Mean Square Error Loss."""

    def __init__(self, loss_weight: float = 1., reduction: str = 'none'):
        super().__init__(loss_weight=loss_weight)
        self.reduction = reduction

    def _forward(self, cls_score: torch.Tensor, label: torch.Tensor,
                 **kwargs) -> torch.Tensor:
        """Forward function.

        Args:
            cls_score (torch.Tensor): The class score.
            label (torch.Tensor): The ground truth label.
            kwargs: Any keyword argument to be used to calculate
                MeanSquareError loss.

        Returns:
            torch.Tensor: The returned MeanSquareError loss.
        """
        if cls_score.size() == label.size():
            assert len(kwargs) == 0, \
                ('For now, no extra args are supported for soft label, '
                 f'but get {kwargs}')

            loss_cls = F.mse_loss(cls_score, label, reduction=self.reduction)
        return loss_cls
