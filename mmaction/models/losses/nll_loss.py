import torch.nn as nn
import torch.nn.functional as F

from ..registry import LOSSES


@LOSSES.register_module
class NLLLoss(nn.Module):
    """NLL Loss.

    It will calculate NLL loss given cls_score and label.
    """

    def forward(self, cls_score, label):
        loss_cls = F.nll_loss(cls_score, label)
        return loss_cls
