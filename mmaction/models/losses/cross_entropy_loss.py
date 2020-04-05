import torch.nn as nn
import torch.nn.functional as F

from ..registry import LOSSES


@LOSSES.register_module
class CrossEntropyLoss(nn.Module):
    """Cross Entropy Loss

    It will calculate cross_entropy loss given cls_score and label.
    """

    def forward(self, cls_score, label):
        loss_cls = F.cross_entropy(cls_score, label)
        return loss_cls
