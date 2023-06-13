# Copyright (c) OpenMMLab. All rights reserved.
from abc import abstractproperty

import torch

from mmaction.structures import bbox_target


def test_bbox_target():
    pos_bboxes = torch.tensor([[0.072, 0.47, 0.84, 0.898],
                               [0.23, 0.215, 0.781, 0.534],
                               [0.195, 0.128, 0.643, 0.944],
                               [0.236, 0.189, 0.689, 0.74]])
    neg_bboxes = torch.tensor([[0.375, 0.371, 0.726, 0.804],
                               [0.024, 0.398, 0.776, 0.719]])
    pos_gt_labels = torch.tensor([[0., 0., 1., 0.], [0., 0., 0., 1.],
                                  [0., 1., 0., 0.], [0., 1., 0., 0.]])
    cfg = abstractproperty()
    cfg.pos_weight = 0.8
    labels, label_weights = bbox_target([pos_bboxes], [neg_bboxes],
                                        [pos_gt_labels], cfg)
    assert torch.all(
        torch.isclose(
            labels,
            torch.tensor([[0., 0., 1., 0.], [0., 0., 0., 1.], [0., 1., 0., 0.],
                          [0., 1., 0., 0.], [0., 0., 0., 0.], [0., 0., 0.,
                                                               0.]])))
    assert torch.all(
        torch.isclose(label_weights, torch.tensor([0.8] * 4 + [1.0] * 2)))
