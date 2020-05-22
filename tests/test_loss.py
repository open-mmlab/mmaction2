import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.testing import assert_array_almost_equal

from mmaction.models import (BCELossWithLogits, BinaryLogisticRegressionLoss,
                             BMNLoss, CrossEntropyLoss, NLLLoss)


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


def test_bmn_loss():
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

    # test pem_reg_loss
    seed = 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    pred_bm_reg = torch.tensor([[0.1, 0.99], [0.5, 0.4]])
    gt_iou_map = torch.tensor([[0, 1.], [0, 1.]])
    mask = torch.tensor([[0.1, 0.4], [0.4, 0.1]])
    output_pem_reg_loss = bmn_loss.pem_reg_loss(pred_bm_reg, gt_iou_map, mask)
    assert_array_almost_equal(
        output_pem_reg_loss.numpy(), np.array([0.2140]), decimal=4)

    # test pem_cls_loss
    pred_bm_cls = torch.tensor([[0.1, 0.99], [0.95, 0.2]])
    gt_iou_map = torch.tensor([[0., 1.], [0., 1.]])
    mask = torch.tensor([[0.1, 0.4], [0.4, 0.1]])
    output_pem_cls_loss = bmn_loss.pem_cls_loss(pred_bm_cls, gt_iou_map, mask)
    assert_array_almost_equal(
        output_pem_cls_loss.numpy(), np.array([1.6137]), decimal=4)

    # test bmn_loss
    pred_bm = torch.tensor([[[[0.1, 0.99], [0.5, 0.4]],
                             [[0.1, 0.99], [0.95, 0.2]]]])
    pred_start = torch.tensor([[0.9, 0.1]])
    pred_end = torch.tensor([[0.1, 0.9]])
    gt_iou_map = torch.tensor([[[0., 2.5], [0., 10.]]])
    gt_start = torch.tensor([[1., 0.]])
    gt_end = torch.tensor([[0., 1.]])
    mask = torch.tensor([[0.1, 0.4], [0.4, 0.1]])
    output_loss = bmn_loss(pred_bm, pred_start, pred_end, gt_iou_map, gt_start,
                           gt_end, mask)
    assert_array_almost_equal(
        output_loss[0].numpy(),
        output_tem_loss + 10 * output_pem_reg_loss + output_pem_cls_loss)
    assert_array_almost_equal(output_loss[1].numpy(), output_tem_loss)
    assert_array_almost_equal(output_loss[2].numpy(), output_pem_reg_loss)
    assert_array_almost_equal(output_loss[3].numpy(), output_pem_cls_loss)
