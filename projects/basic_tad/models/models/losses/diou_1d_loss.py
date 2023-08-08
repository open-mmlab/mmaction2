# adapted from basicTAD
import torch
import torch.nn as nn
from mmdet.models.losses import DIoULoss
from mmdet.models.losses.utils import weighted_loss
from torch import Tensor

from mmaction.registry import MODELS
from ..task_modules.segments_ops import segment_overlaps


def zero_out_loss_coordinates_decorator(forward_method):
    def wrapper(self, pred: Tensor, target: Tensor, *args, **kwargs):
        pred = pred.clone()
        pred[:, 1] = pred[:, 1] * 0 + target[:, 1]
        pred[:, 3] = pred[:, 3] * 0 + target[:, 3]
        return forward_method(self, pred, target, *args, **kwargs)

    return wrapper


@weighted_loss
def iou_loss(pred, target, eps=1e-6):
    """IoU loss.

    Computing the IoU loss between a set of predicted segments and target
        segments.
    The loss is calculated as negative log of IoU.
    Args:
        pred (torch.Tensor): Predicted segments of format (start, end),
            shape (n, 2).
        target (torch.Tensor): Corresponding gt segments, shape (n, 2).
        eps (float): Eps to avoid log(0).
    Return:
        torch.Tensor: Loss tensor.
    """
    ious = segment_overlaps(pred, target, is_aligned=True).clamp(min=eps)
    loss = -ious.log()
    return loss


@weighted_loss
def bounded_iou_loss(pred, target, beta=0.2, eps=1e-3):
    """BIoULoss. This is an implementation of paper `Improving Object
    Localization with Fitness NMS and Bounded IoU Loss.

    <https://arxiv.org/abs/1711.00164>`_.
    Args:
        pred (torch.Tensor): Predicted segments.
        target (torch.Tensor): Target segments.
        beta (float): beta parameter in smoothl1.
        eps (float): eps to avoid NaN.
    """
    pred_center = (pred[:, 0] + pred[:, 1]) * 0.5
    pred_interval = pred[:, 1] - pred[:, 0]
    with torch.no_grad():
        target_center = (target[:, 0] + target[:, 1]) * 0.5
        target_interval = target[:, 1] - target[:, 0]

    d_center = target_center - pred_center

    loss_center = 1 - torch.max((target_interval - 2 * d_center.abs()) /
                                (target_interval + 2 * d_center.abs() + eps),
                                torch.zeros_like(d_center))
    loss_interval = 1 - torch.min(target_interval /
                                  (pred_interval + eps), pred_interval /
                                  (target_interval + eps))
    loss_comb = torch.stack([loss_center, loss_interval],
                            dim=-1).view(loss_center.size(0), -1)

    loss = torch.where(loss_comb < beta, 0.5 * loss_comb * loss_comb / beta,
                       loss_comb - 0.5 * beta)
    return loss


@weighted_loss
def giou_loss(pred, target, eps=1e-7):
    r"""`Generalized Intersection over Union: A Metric and A Loss for Bounding
    Box Regression <https://arxiv.org/abs/1902.09630>`_.
    Args:
        pred (torch.Tensor): Predicted segments of format (start, end),
            shape (n, 2).
        target (torch.Tensor): Corresponding gt segments, shape (n, 2).
        eps (float): Eps to avoid log(0).
    Return:
        Tensor: Loss tensor.
    """
    # overlap
    start = torch.max(pred[:, 0], target[:, 0])
    end = torch.min(pred[:, 1], target[:, 1])
    overlap = (end - start).clamp(min=0)

    # union
    ap = pred[:, 1] - pred[:, 0]
    ag = target[:, 1] - target[:, 0]
    union = ap + ag - overlap + eps

    # IoU
    ious = overlap / union

    # enclose area
    enclose_start = torch.min(pred[:, 0], target[:, 0])
    enclose_end = torch.max(pred[:, 1], target[:, 1])
    enclose_interval = (enclose_end - enclose_start).clamp(min=0) + eps

    # GIoU
    gious = ious - (enclose_interval - union) / enclose_interval
    loss = 1 - gious
    return loss


@weighted_loss
def diou_loss(pred, target, eps=1e-7):
    r"""`Implementation of Distance-IoU Loss: Faster and Better
    Learning for Bounding Box Regression, https://arxiv.org/abs/1911.08287`_.
    Code is modified from https://github.com/Zzh-tju/DIoU.
    Args:
        pred (Tensor): Predicted segments of format (start, end),
            shape (n, 2).
        target (Tensor): Corresponding gt segments, shape (n, 2).
        eps (float): Eps to avoid log(0).
    Return:
        Tensor: Loss tensor.
    """
    # overlap
    start = torch.max(pred[:, 0], target[:, 0])
    end = torch.min(pred[:, 1], target[:, 1])
    overlap = (end - start).clamp(min=0)

    # union
    ap = pred[:, 1] - pred[:, 0]
    ag = target[:, 1] - target[:, 0]
    union = ap + ag - overlap + eps

    # IoU
    ious = overlap / union

    # enclose area
    enclose_start = torch.min(pred[:, 0], target[:, 0])
    enclose_end = torch.max(pred[:, 1], target[:, 1])
    enclose_interval = (enclose_end - enclose_start).clamp(min=0)
    c2 = enclose_interval ** 2 + eps

    pred_center = (pred[:, 0] + pred[:, 1]) * 0.5
    target_center = (target[:, 0] + target[:, 1]) * 0.5
    rho2 = (target_center - pred_center) ** 2

    # DIoU
    dious = ious - rho2 / c2
    loss = 1 - dious
    return loss


@MODELS.register_module()
class IoULoss(nn.Module):
    """IoULoss.

    Computing the IoU loss between a set of predicted segments and target
    segments.
    Args:
        eps (float): Eps to avoid log(0).
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss.
    """

    def __init__(self, eps=1e-6, reduction='mean', loss_weight=1.0):
        super(IoULoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None. Options are "none", "mean" and "sum".
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if (weight is not None) and (not torch.any(weight > 0)) and (
                reduction != 'none'):
            return (pred * weight).sum()  # 0
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # iou_loss of shape (n,)
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * iou_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss


@MODELS.register_module()
class BoundedIoULoss(nn.Module):

    def __init__(self, beta=0.2, eps=1e-3, reduction='mean', loss_weight=1.0):
        super(BoundedIoULoss, self).__init__()
        self.beta = beta
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        if weight is not None and not torch.any(weight > 0):
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = self.loss_weight * bounded_iou_loss(
            pred,
            target,
            weight,
            beta=self.beta,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss


@MODELS.register_module()
class GIoULoss(nn.Module):

    def __init__(self, eps=1e-6, reduction='mean', loss_weight=1.0):
        super(GIoULoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        if weight is not None and not torch.any(weight > 0):
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # giou_loss of shape (n,)
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * giou_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss


@MODELS.register_module()
class DIoU1DLoss(DIoULoss):
    @zero_out_loss_coordinates_decorator
    def forward(self, pred: Tensor, target: Tensor, *args, **kwargs) -> Tensor:
        return super().forward(pred, target, *args, **kwargs)
