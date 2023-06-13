# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
from mmengine import ConfigDict

from mmaction.models import OHEMHingeLoss, SSNLoss


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
