# Copyright (c) OpenMMLab. All rights reserved.
"""Adapted from https://github.com/Alvin-Zeng/DRN/"""

import torch
import torchvision
from torch import nn

INF = 100000000


def SigmoidFocalLoss(alpha, gamma):

    def loss_fn(inputs, targets):
        loss = torchvision.ops.sigmoid_focal_loss(
            inputs=inputs,
            targets=targets,
            alpha=alpha,
            gamma=gamma,
            reduction='sum')
        return loss

    return loss_fn


def IOULoss():

    def loss_fn(pred, target):
        pred_left = pred[:, 0]
        pred_right = pred[:, 1]

        target_left = target[:, 0]
        target_right = target[:, 1]

        intersect = torch.min(pred_right, target_right) + torch.min(
            pred_left, target_left)
        target_area = target_left + target_right
        pred_area = pred_left + pred_right
        union = target_area + pred_area - intersect

        losses = -torch.log((intersect + 1e-8) / (union + 1e-8))
        return losses.mean()

    return loss_fn


class FCOSLossComputation(object):
    """This class computes the FCOS losses."""

    def __init__(self, focal_alpha, focal_gamma):
        self.cls_loss_fn = SigmoidFocalLoss(focal_alpha, focal_gamma)
        self.box_reg_loss_fn = IOULoss()
        self.centerness_loss_fn = nn.BCEWithLogitsLoss()
        self.iou_loss_fn = nn.SmoothL1Loss()

    def prepare_targets(self, points, targets):
        object_sizes_of_interest = [
            [-1, 6],
            [5.6, 11],
            [11, INF],
        ]
        expanded_object_sizes_of_interest = []
        for idx, points_per_level in enumerate(points):
            object_sizes_of_interest_per_level = \
                points_per_level.new_tensor(object_sizes_of_interest[idx])
            expanded_object_sizes_of_interest.append(
                object_sizes_of_interest_per_level[None].expand(
                    len(points_per_level), -1))

        expanded_object_sizes_of_interest = torch.cat(
            expanded_object_sizes_of_interest, dim=0)
        num_points_per_level = [
            len(points_per_level) for points_per_level in points
        ]
        points_all_level = torch.cat(points, dim=0)
        labels, reg_targets = self.compute_targets_for_locations(
            points_all_level, targets, expanded_object_sizes_of_interest)

        for i in range(len(labels)):
            labels[i] = torch.split(labels[i], num_points_per_level, dim=0)
            reg_targets[i] = torch.split(
                reg_targets[i], num_points_per_level, dim=0)

        labels_level_first = []
        reg_targets_level_first = []
        for level in range(len(points)):
            labels_level_first.append(
                torch.cat([labels_per_im[level] for labels_per_im in labels],
                          dim=0))
            reg_targets_level_first.append(
                torch.cat([
                    reg_targets_per_im[level]
                    for reg_targets_per_im in reg_targets
                ],
                          dim=0))

        return labels_level_first, reg_targets_level_first

    def compute_targets_for_locations(self, locations, targets,
                                      object_sizes_of_interest):
        labels = []
        reg_targets = []
        ts = locations

        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            bboxes = targets_per_im * 32

            left = ts[:, None] - bboxes[None, 0]
            right = bboxes[None, 1] - ts[:, None]
            reg_targets_per_im = torch.cat([left, right], dim=1)

            is_in_boxes = reg_targets_per_im.min(dim=1)[0] > 0
            max_reg_targets_per_im = reg_targets_per_im.max(dim=1)[0]
            is_cared_in_the_level = \
                (max_reg_targets_per_im >= object_sizes_of_interest[:, 0]) & \
                (max_reg_targets_per_im <= object_sizes_of_interest[:, 1])

            locations_to_gt_area = bboxes[1] - bboxes[0]
            locations_to_gt_area = locations_to_gt_area.repeat(
                len(locations), 1)
            locations_to_gt_area[is_in_boxes == 0] = INF
            locations_to_gt_area[is_cared_in_the_level == 0] = INF

            _ = locations_to_gt_area.min(dim=1)
            locations_to_min_area, locations_to_gt_inds = _

            labels_per_im = reg_targets_per_im.new_ones(
                len(reg_targets_per_im))
            labels_per_im[locations_to_min_area == INF] = 0

            labels.append(labels_per_im)
            reg_targets.append(reg_targets_per_im)

        return labels, reg_targets

    def __call__(self,
                 locations,
                 box_cls,
                 box_regression,
                 targets,
                 iou_scores,
                 is_first_stage=True):
        N = box_cls[0].size(0)
        num_classes = box_cls[0].size(1)
        labels, reg_targets = self.prepare_targets(locations, targets)

        box_cls_flatten = []
        box_regression_flatten = []
        # centerness_flatten = []
        labels_flatten = []
        reg_targets_flatten = []

        for idx in range(len(labels)):
            box_cls_flatten.append(box_cls[idx].permute(0, 2, 1).reshape(
                -1, num_classes))
            box_regression_flatten.append(box_regression[idx].permute(
                0, 2, 1).reshape(-1, 2))
            labels_flatten.append(labels[idx].reshape(-1))
            reg_targets_flatten.append(reg_targets[idx].reshape(-1, 2))

        if not is_first_stage:
            # [batch, 56, 2]
            merged_box_regression = torch.cat(
                box_regression, dim=-1).transpose(2, 1)
            # [56]
            merged_locations = torch.cat(locations, dim=0)
            # [batch, 56]
            full_locations = merged_locations[None, :].expand(
                merged_box_regression.size(0), -1).contiguous()
            pred_start = full_locations - merged_box_regression[:, :, 0]
            pred_end = full_locations + merged_box_regression[:, :, 1]
            # [batch, 56, 2]
            predictions = torch.cat(
                [pred_start.unsqueeze(-1),
                 pred_end.unsqueeze(-1)], dim=-1) / 32
            # TODO: make sure the predictions are legal. (e.g. start < end)
            predictions.clamp_(min=0, max=1)
            # gt: [batch, 2]
            gt_box = targets[:, None, :]

            iou_target = segment_tiou(predictions, gt_box)
            iou_pred = torch.cat(iou_scores, dim=-1).squeeze().sigmoid()
            iou_pos_ind = iou_target > 0.9
            pos_iou_target = iou_target[iou_pos_ind]

            pos_iou_pred = iou_pred[iou_pos_ind]

            if iou_pos_ind.sum().item() == 0:
                iou_loss = torch.tensor([0.]).to(iou_pos_ind.device)
            else:
                iou_loss = self.iou_loss_fn(pos_iou_pred, pos_iou_target)

        box_cls_flatten = torch.cat(box_cls_flatten, dim=0)
        box_regression_flatten = torch.cat(box_regression_flatten, dim=0)
        labels_flatten = torch.cat(labels_flatten, dim=0)
        reg_targets_flatten = torch.cat(reg_targets_flatten, dim=0)

        pos_inds = torch.nonzero(labels_flatten > 0).squeeze(1)
        cls_loss = self.cls_loss_fn(
            box_cls_flatten, labels_flatten.unsqueeze(1)) / (
                pos_inds.numel() + N)  # add N to avoid dividing by a zero

        box_regression_flatten = box_regression_flatten[pos_inds]
        reg_targets_flatten = reg_targets_flatten[pos_inds]

        if pos_inds.numel() > 0:
            reg_loss = self.box_reg_loss_fn(
                box_regression_flatten,
                reg_targets_flatten,
            )
        else:
            reg_loss = box_regression_flatten.sum()

        if not is_first_stage:
            return cls_loss, reg_loss, iou_loss

        return cls_loss, reg_loss, torch.tensor([0.]).to(cls_loss.device)


def segment_tiou(box_a, box_b):

    # gt: [batch, 1, 2], detections: [batch, 56, 2]
    # calculate interaction
    inter_max_xy = torch.min(box_a[:, :, -1], box_b[:, :, -1])
    inter_min_xy = torch.max(box_a[:, :, 0], box_b[:, :, 0])
    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)

    # calculate union
    union_max_xy = torch.max(box_a[:, :, -1], box_b[:, :, -1])
    union_min_xy = torch.min(box_a[:, :, 0], box_b[:, :, 0])
    union = torch.clamp((union_max_xy - union_min_xy), min=0)

    iou = inter / (union + 1e-6)

    return iou


def make_fcos_loss_evaluator(focal_alpha, focal_gamma):
    loss_evaluator = FCOSLossComputation(focal_alpha, focal_gamma)
    return loss_evaluator
