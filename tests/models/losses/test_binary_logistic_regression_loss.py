# Copyright (c) OpenMMLab. All rights reserved.
import torch
from numpy.testing import assert_array_almost_equal

from mmaction.models import BinaryLogisticRegressionLoss, BMNLoss


def test_binary_logistic_regression_loss():
    bmn_loss = BMNLoss()

    # test tem_loss
    pred_start = torch.tensor([0.9, 0.1])
    pred_end = torch.tensor([0.1, 0.9])
    gt_start = torch.tensor([1., 0.])
    gt_end = torch.tensor([0., 1.])
    output_tem_loss = bmn_loss.tem_loss(pred_start, pred_end, gt_start, gt_end)
    binary_logistic_regression_loss = BinaryLogisticRegressionLoss()
    assert_loss = (
        binary_logistic_regression_loss(pred_start, gt_start) +
        binary_logistic_regression_loss(pred_end, gt_end))
    assert_array_almost_equal(
        output_tem_loss.numpy(), assert_loss.numpy(), decimal=4)
