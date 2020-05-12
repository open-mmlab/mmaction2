import torch
import torch.nn as nn
import torch.nn.functional as F

from mmaction.models import BCELossWithLogits, CrossEntropyLoss, NLLLoss


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
