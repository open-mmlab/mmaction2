# Copyright (c) Facebook, Inc. and its affiliates. Modified by Jialian Wu
# from https://github.com/facebookresearch/Detic/blob/main/detic/modeling
# /roi_heads/detic_fast_rcnn.py
import fvcore.nn.weight_init as weight_init
import torch
from detectron2.config import configurable
from detectron2.layers import ShapeSpec, cat, nonzero_tuple
from detectron2.modeling.roi_heads.fast_rcnn import (FastRCNNOutputLayers,
                                                     _log_classification_stats)
from fvcore.nn import giou_loss, smooth_l1_loss
from torch import nn
from torch.nn import functional as F

__all__ = ['GRiTFastRCNNOutputLayers']


class GRiTFastRCNNOutputLayers(FastRCNNOutputLayers):

    @configurable
    def __init__(
        self,
        input_shape: ShapeSpec,
        **kwargs,
    ):
        super().__init__(
            input_shape=input_shape,
            **kwargs,
        )

        input_size = input_shape.channels * \
            (input_shape.width or 1) * (input_shape.height or 1)

        self.bbox_pred = nn.Sequential(
            nn.Linear(input_size, input_size), nn.ReLU(inplace=True),
            nn.Linear(input_size, 4))
        weight_init.c2_xavier_fill(self.bbox_pred[0])
        nn.init.normal_(self.bbox_pred[-1].weight, std=0.001)
        nn.init.constant_(self.bbox_pred[-1].bias, 0)

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        return ret

    def losses(self, predictions, proposals):
        scores, proposal_deltas = predictions
        gt_classes = (
            cat([p.gt_classes for p in proposals], dim=0)
            if len(proposals) else torch.empty(0))
        num_classes = self.num_classes
        _log_classification_stats(scores, gt_classes)

        if len(proposals):
            proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals],
                                 dim=0)  # Nx4
            assert not proposal_boxes.requires_grad, \
                'Proposals should not require gradients!'
            gt_boxes = cat(
                [(p.gt_boxes if p.has('gt_boxes') else p.proposal_boxes).tensor
                 for p in proposals],
                dim=0,
            )
        else:
            proposal_boxes = gt_boxes = torch.empty(
                (0, 4), device=proposal_deltas.device)

        loss_cls = self.softmax_cross_entropy_loss(scores, gt_classes)
        return {
            'loss_cls':
            loss_cls,
            'loss_box_reg':
            self.box_reg_loss(
                proposal_boxes,
                gt_boxes,
                proposal_deltas,
                gt_classes,
                num_classes=num_classes)
        }

    def softmax_cross_entropy_loss(self, pred_class_logits, gt_classes):
        if pred_class_logits.numel() == 0:
            return pred_class_logits.new_zeros([1])[0]

        loss = F.cross_entropy(pred_class_logits, gt_classes, reduction='mean')
        return loss

    def box_reg_loss(self,
                     proposal_boxes,
                     gt_boxes,
                     pred_deltas,
                     gt_classes,
                     num_classes=-1):
        num_classes = num_classes if num_classes > 0 else self.num_classes
        box_dim = proposal_boxes.shape[1]
        fg_inds = nonzero_tuple((gt_classes >= 0)
                                & (gt_classes < num_classes))[0]
        if pred_deltas.shape[1] == box_dim:
            fg_pred_deltas = pred_deltas[fg_inds]
        else:
            fg_pred_deltas = pred_deltas.view(-1, self.num_classes,
                                              box_dim)[fg_inds,
                                                       gt_classes[fg_inds]]

        if self.box_reg_loss_type == 'smooth_l1':
            gt_pred_deltas = self.box2box_transform.get_deltas(
                proposal_boxes[fg_inds],
                gt_boxes[fg_inds],
            )
            loss_box_reg = smooth_l1_loss(
                fg_pred_deltas,
                gt_pred_deltas,
                self.smooth_l1_beta,
                reduction='sum')
        elif self.box_reg_loss_type == 'giou':
            fg_pred_boxes = self.box2box_transform.apply_deltas(
                fg_pred_deltas, proposal_boxes[fg_inds])
            loss_box_reg = giou_loss(
                fg_pred_boxes, gt_boxes[fg_inds], reduction='sum')
        else:
            raise ValueError(
                f"Invalid bbox reg loss type '{self.box_reg_loss_type}'")
        return loss_box_reg / max(gt_classes.numel(), 1.0)

    def predict_probs(self, predictions, proposals):
        scores = predictions[0]
        num_inst_per_image = [len(p) for p in proposals]
        probs = F.softmax(scores, dim=-1)
        return probs.split(num_inst_per_image, dim=0)

    def forward(self, x):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        scores = []

        cls_scores = self.cls_score(x)
        scores.append(cls_scores)
        scores = torch.cat(scores, dim=1)

        proposal_deltas = self.bbox_pred(x)
        return scores, proposal_deltas
