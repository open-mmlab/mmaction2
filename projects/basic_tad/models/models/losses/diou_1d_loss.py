# adapted from basicTAD
from mmdet.models.losses import DIoULoss
from torch import Tensor

from mmaction.registry import MODELS


def zero_out_loss_coordinates_decorator(forward_method):
    def wrapper(self, pred: Tensor, target: Tensor, *args, **kwargs):
        pred = pred.clone()
        pred[:, 1] = pred[:, 1] * 0 + target[:, 1]
        pred[:, 3] = pred[:, 3] * 0 + target[:, 3]
        return forward_method(self, pred, target, *args, **kwargs)

    return wrapper


@MODELS.register_module()
class DIoU1DLoss(DIoULoss):
    @zero_out_loss_coordinates_decorator
    def forward(self, pred: Tensor, target: Tensor, *args, **kwargs) -> Tensor:
        return super().forward(pred, target, *args, **kwargs)
