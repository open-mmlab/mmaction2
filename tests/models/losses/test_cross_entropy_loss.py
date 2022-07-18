# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
from numpy.testing import assert_almost_equal

from mmaction.models import BCELossWithLogits, CrossEntropyLoss


def test_bce_loss_with_logits():
    cls_scores = torch.rand((3, 4))
    gt_labels = torch.rand((3, 4))

    bce_loss_with_logits = BCELossWithLogits()
    output_loss = bce_loss_with_logits(cls_scores, gt_labels)
    assert torch.equal(
        output_loss, F.binary_cross_entropy_with_logits(cls_scores, gt_labels))

    weight = torch.rand(4)
    class_weight = weight.numpy().tolist()
    bce_loss_with_logits = BCELossWithLogits(class_weight=class_weight)
    output_loss = bce_loss_with_logits(cls_scores, gt_labels)
    assert torch.equal(
        output_loss,
        F.binary_cross_entropy_with_logits(
            cls_scores, gt_labels, weight=weight))


def test_cross_entropy_loss():
    cls_scores = torch.rand((3, 4))
    hard_gt_labels = torch.LongTensor([0, 1, 2]).squeeze()
    soft_gt_labels = torch.FloatTensor([[1, 0, 0, 0], [0, 1, 0, 0],
                                        [0, 0, 1, 0]]).squeeze()

    # hard label without weight
    cross_entropy_loss = CrossEntropyLoss()
    output_loss = cross_entropy_loss(cls_scores, hard_gt_labels)
    assert torch.equal(output_loss, F.cross_entropy(cls_scores,
                                                    hard_gt_labels))

    # hard label with class weight
    weight = torch.rand(4)
    class_weight = weight.numpy().tolist()
    cross_entropy_loss = CrossEntropyLoss(class_weight=class_weight)
    output_loss = cross_entropy_loss(cls_scores, hard_gt_labels)
    assert torch.equal(
        output_loss,
        F.cross_entropy(cls_scores, hard_gt_labels, weight=weight))

    # soft label without class weight
    cross_entropy_loss = CrossEntropyLoss()
    output_loss = cross_entropy_loss(cls_scores, soft_gt_labels)
    assert_almost_equal(
        output_loss.numpy(),
        F.cross_entropy(cls_scores, hard_gt_labels).numpy(),
        decimal=4)

    # soft label with class weight
    cross_entropy_loss = CrossEntropyLoss(class_weight=class_weight)
    output_loss = cross_entropy_loss(cls_scores, soft_gt_labels)
    assert_almost_equal(
        output_loss.numpy(),
        F.cross_entropy(cls_scores, hard_gt_labels, weight=weight).numpy(),
        decimal=4)
