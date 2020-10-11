import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv import ConfigDict
from numpy.testing import assert_array_almost_equal
from torch.autograd import Variable

from mmaction.models import (BCELossWithLogits, BinaryLogisticRegressionLoss,
                             BMNLoss, CrossEntropyLoss, HVULoss, NLLLoss,
                             OHEMHingeLoss, SSNLoss)


def test_hvu_loss():
    pred = torch.tensor([[-1.0525, -0.7085, 0.1819, -0.8011],
                         [0.1555, -1.5550, 0.5586, 1.9746]])
    gt = torch.tensor([[1., 0., 0., 0.], [0., 0., 1., 1.]])
    mask = torch.tensor([[1., 1., 0., 0.], [0., 0., 1., 1.]])
    category_mask = torch.tensor([[1., 0.], [0., 1.]])
    categories = ['action', 'scene']
    category_nums = (2, 2)
    category_loss_weights = (1, 1)
    loss_all_nomask_sum = HVULoss(
        categories=categories,
        category_nums=category_nums,
        category_loss_weights=category_loss_weights,
        loss_type='all',
        with_mask=False,
        reduction='sum')
    loss = loss_all_nomask_sum(pred, gt, mask, category_mask)
    loss1 = F.binary_cross_entropy_with_logits(pred, gt, reduction='none')
    loss1 = torch.sum(loss1, dim=1)
    assert torch.eq(loss['loss_cls'], torch.mean(loss1))

    loss_all_mask = HVULoss(
        categories=categories,
        category_nums=category_nums,
        category_loss_weights=category_loss_weights,
        loss_type='all',
        with_mask=True)
    loss = loss_all_mask(pred, gt, mask, category_mask)
    loss1 = F.binary_cross_entropy_with_logits(pred, gt, reduction='none')
    loss1 = torch.sum(loss1 * mask, dim=1) / torch.sum(mask, dim=1)
    loss1 = torch.mean(loss1)
    assert torch.eq(loss['loss_cls'], loss1)

    loss_ind_mask = HVULoss(
        categories=categories,
        category_nums=category_nums,
        category_loss_weights=category_loss_weights,
        loss_type='individual',
        with_mask=True)
    loss = loss_ind_mask(pred, gt, mask, category_mask)
    action_loss = F.binary_cross_entropy_with_logits(pred[:1, :2], gt[:1, :2])
    scene_loss = F.binary_cross_entropy_with_logits(pred[1:, 2:], gt[1:, 2:])
    loss1 = (action_loss + scene_loss) / 2
    assert torch.eq(loss['loss_cls'], loss1)

    loss_ind_nomask_sum = HVULoss(
        categories=categories,
        category_nums=category_nums,
        category_loss_weights=category_loss_weights,
        loss_type='individual',
        with_mask=False,
        reduction='sum')
    loss = loss_ind_nomask_sum(pred, gt, mask, category_mask)
    action_loss = F.binary_cross_entropy_with_logits(
        pred[:, :2], gt[:, :2], reduction='none')
    action_loss = torch.sum(action_loss, dim=1)
    action_loss = torch.mean(action_loss)

    scene_loss = F.binary_cross_entropy_with_logits(
        pred[:, 2:], gt[:, 2:], reduction='none')
    scene_loss = torch.sum(scene_loss, dim=1)
    scene_loss = torch.mean(scene_loss)

    loss1 = (action_loss + scene_loss) / 2
    assert torch.eq(loss['loss_cls'], loss1)


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


def test_ohem_hinge_loss():
    # test normal case
    pred = torch.tensor([[
        0.5161, 0.5228, 0.7748, 0.0573, 0.1113, 0.8862, 0.1752, 0.9448, 0.0253,
        0.1009, 0.4371, 0.2232, 0.0412, 0.3487, 0.3350, 0.9294, 0.7122, 0.3072,
        0.2942, 0.7679
    ]],
                        requires_grad=True)
    gt = torch.tensor([8])
    num_video = 1
    loss = OHEMHingeLoss.apply(pred, gt, 1, 1.0, num_video)
    assert_array_almost_equal(
        loss.detach().numpy(), np.array([0.0552]), decimal=4)
    loss.backward(Variable(torch.ones([1])))
    assert_array_almost_equal(
        np.array(pred.grad),
        np.array([[
            0., 0., 0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0.
        ]]),
        decimal=4)

    # test error case
    with pytest.raises(ValueError):
        gt = torch.tensor([8, 10])
        loss = OHEMHingeLoss.apply(pred, gt, 1, 1.0, num_video)


def test_ssn_loss():
    ssn_loss = SSNLoss()

    # test activity_loss
    activity_score = torch.rand((8, 21))
    labels = torch.LongTensor([8] * 8).squeeze()
    activity_indexer = torch.tensor([0, 7])
    output_activity_loss = ssn_loss.activity_loss(activity_score, labels,
                                                  activity_indexer)
    assert torch.equal(
        output_activity_loss,
        F.cross_entropy(activity_score[activity_indexer, :],
                        labels[activity_indexer]))

    # test completeness_loss
    completeness_score = torch.rand((8, 20), requires_grad=True)
    labels = torch.LongTensor([8] * 8).squeeze()
    completeness_indexer = torch.tensor([0, 1, 2, 3, 4, 5, 6])
    positive_per_video = 1
    incomplete_per_video = 6
    output_completeness_loss = ssn_loss.completeness_loss(
        completeness_score, labels, completeness_indexer, positive_per_video,
        incomplete_per_video)

    pred = completeness_score[completeness_indexer, :]
    gt = labels[completeness_indexer]
    pred_dim = pred.size(1)
    pred = pred.view(-1, positive_per_video + incomplete_per_video, pred_dim)
    gt = gt.view(-1, positive_per_video + incomplete_per_video)
    # yapf:disable
    positive_pred = pred[:, :positive_per_video, :].contiguous().view(-1, pred_dim)  # noqa:E501
    incomplete_pred = pred[:, positive_per_video:, :].contiguous().view(-1, pred_dim)  # noqa:E501
    # yapf:enable
    ohem_ratio = 0.17
    positive_loss = OHEMHingeLoss.apply(
        positive_pred, gt[:, :positive_per_video].contiguous().view(-1), 1,
        1.0, positive_per_video)
    incomplete_loss = OHEMHingeLoss.apply(
        incomplete_pred, gt[:, positive_per_video:].contiguous().view(-1), -1,
        ohem_ratio, incomplete_per_video)
    num_positives = positive_pred.size(0)
    num_incompletes = int(incomplete_pred.size(0) * ohem_ratio)
    assert_loss = ((positive_loss + incomplete_loss) /
                   float(num_positives + num_incompletes))
    assert torch.equal(output_completeness_loss, assert_loss)

    # test reg_loss
    bbox_pred = torch.rand((8, 20, 2))
    labels = torch.LongTensor([8] * 8).squeeze()
    bbox_targets = torch.rand((8, 2))
    regression_indexer = torch.tensor([0])
    output_reg_loss = ssn_loss.classwise_regression_loss(
        bbox_pred, labels, bbox_targets, regression_indexer)

    pred = bbox_pred[regression_indexer, :, :]
    gt = labels[regression_indexer]
    reg_target = bbox_targets[regression_indexer, :]
    class_idx = gt.data - 1
    classwise_pred = pred[:, class_idx, :]
    classwise_reg_pred = torch.cat((torch.diag(classwise_pred[:, :, 0]).view(
        -1, 1), torch.diag(classwise_pred[:, :, 1]).view(-1, 1)),
                                   dim=1)
    assert torch.equal(
        output_reg_loss,
        F.smooth_l1_loss(classwise_reg_pred.view(-1), reg_target.view(-1)) * 2)

    # test ssn_loss
    proposal_type = torch.tensor([[0, 1, 1, 1, 1, 1, 1, 2]])
    train_cfg = ConfigDict(
        dict(
            ssn=dict(
                sampler=dict(
                    num_per_video=8,
                    positive_ratio=1,
                    background_ratio=1,
                    incomplete_ratio=6,
                    add_gt_as_proposals=True),
                loss_weight=dict(comp_loss_weight=0.1, reg_loss_weight=0.1))))
    output_loss = ssn_loss(activity_score, completeness_score, bbox_pred,
                           proposal_type, labels, bbox_targets, train_cfg)
    assert torch.equal(output_loss['loss_activity'], output_activity_loss)
    assert torch.equal(output_loss['loss_completeness'],
                       output_completeness_loss * 0.1)
    assert torch.equal(output_loss['loss_reg'], output_reg_loss * 0.1)
