# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F


def bbox_target(pos_bboxes_list, neg_bboxes_list, gt_labels, cfg):
    """Generate classification targets for bboxes.

    Args:
        pos_bboxes_list (list[Tensor]): Positive bboxes list.
        neg_bboxes_list (list[Tensor]): Negative bboxes list.
        gt_labels (list[Tensor]): Groundtruth classification label list.
        cfg (Config): RCNN config.

    Returns:
        (Tensor, Tensor): Label and label_weight for bboxes.
    """
    labels, label_weights = [], []
    pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight

    assert len(pos_bboxes_list) == len(neg_bboxes_list) == len(gt_labels)
    length = len(pos_bboxes_list)

    for i in range(length):
        pos_bboxes = pos_bboxes_list[i]
        neg_bboxes = neg_bboxes_list[i]
        gt_label = gt_labels[i]

        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg
        label = F.pad(gt_label, (0, 0, 0, num_neg))
        label_weight = pos_bboxes.new_zeros(num_samples)
        label_weight[:num_pos] = pos_weight
        label_weight[-num_neg:] = 1.

        labels.append(label)
        label_weights.append(label_weight)

    labels = torch.cat(labels, 0)
    label_weights = torch.cat(label_weights, 0)
    return labels, label_weights
