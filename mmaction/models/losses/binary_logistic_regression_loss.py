import torch
import torch.nn as nn

from ..registry import LOSSES


def binary_logistic_regression_loss(reg_score,
                                    label,
                                    threshold=0.5,
                                    ratio_range=(1.05, 21),
                                    eps=1e-5):
    """Binary Logistic Regression Loss."""
    label = label.view(-1).to(reg_score.device)
    reg_score = reg_score.contiguous().view(-1)

    pmask = (label > threshold).float().to(reg_score.device)
    num_positive = max(torch.sum(pmask), 1)
    num_entries = len(label)
    ratio = num_entries / num_positive
    # clip ratio value between ratio_range
    ratio = min(max(ratio, ratio_range[0]), ratio_range[1])

    coef_0 = 0.5 * ratio / (ratio - 1)
    coef_1 = 0.5 * ratio
    loss = coef_1 * pmask * torch.log(reg_score + eps) + coef_0 * (
        1.0 - pmask) * torch.log(1.0 - reg_score + eps)
    loss = -torch.mean(loss)
    return loss


@LOSSES.register_module()
class BinaryLogisticRegressionLoss(nn.Module):
    """Binary Logistic Regression Loss.

    It will calculate binary logistic regression loss given reg_score and
    label.
    """

    def forward(self,
                reg_score,
                label,
                threshold=0.5,
                ratio_range=(1.05, 21),
                eps=1e-5):
        """Calculate Binary Logistic Regression Loss.

        Args:
                reg_score (torch.Tensor): Predicted score by model.
                gt_label (torch.Tensor): Groundtruth labels.
                threshold (float): Threshold for positive instances.
                    Default: 0.5.
                ratio_range (tuple): Lower bound and upper bound for ratio.
                    Default: (1.05, 21)
                eps (float): Epsilon for small value. Default: 1e-5.

        Returns:
                torch.Tensor: Returned binary logistic loss.
        """

        return binary_logistic_regression_loss(reg_score, label, threshold,
                                               ratio_range, eps)
