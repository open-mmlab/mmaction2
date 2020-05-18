import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.testing import assert_array_almost_equal

from mmaction.models import (BCELossWithLogits, BinaryLogisticRegressionLoss,
                             CrossEntropyLoss, NLLLoss)


def test_cross_entropy_loss():
    cls_scores = torch.rand((3, 4))
    gt_labels = torch.LongTensor([2] * 3).squeeze()

    cross_entropy_loss = CrossEntropyLoss()
    output_loss = cross_entropy_loss(cls_scores, gt_labels)
    assert torch.equal(output_loss, F.cross_entropy(cls_scores, gt_labels))


def test_bce_loss_with_logits():
    cls_scores = torch.rand((3, 4))
    gt_labels = torch.rand((3, 4))
    bce_loss_with_logits = BCELossWithLogits()
    output_loss = bce_loss_with_logits(cls_scores, gt_labels)
    assert torch.equal(
        output_loss, F.binary_cross_entropy_with_logits(cls_scores, gt_labels))


def test_nll_loss():
    cls_scores = torch.randn(3, 3)
    gt_labels = torch.tensor([0, 2, 1]).squeeze()

    sm = nn.Softmax(dim=0)
    nll_loss = NLLLoss()
    cls_score_log = torch.log(sm(cls_scores))
    output_loss = nll_loss(cls_score_log, gt_labels)
    assert torch.equal(output_loss, F.nll_loss(cls_score_log, gt_labels))


def test_binary_logistic_loss():
    binary_logistic_regression_loss = BinaryLogisticRegressionLoss()
    reg_score = torch.tensor([0., 1.])
    label = torch.tensor([0., 1.])
    output_loss = binary_logistic_regression_loss(reg_score, label, 0.5)
    assert_array_almost_equal(output_loss.numpy(), np.array([0.]), decimal=4)

    reg_score = torch.tensor([0.3, 0.9])
    label = torch.tensor([0., 1.])
    output_loss = binary_logistic_regression_loss(reg_score, label, 0.5)
    assert_array_almost_equal(
        output_loss.numpy(), np.array([0.231]), decimal=4)
