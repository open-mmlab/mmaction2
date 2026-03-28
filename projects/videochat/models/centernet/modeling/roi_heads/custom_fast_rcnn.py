# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved Part
# of the code is from https://github.com/tztztztztz/eql.detectron2/blob
# /master/projects/EQL/eql/fast_rcnn.py

import torch
from detectron2.layers import ShapeSpec, cat
from detectron2.modeling.roi_heads.fast_rcnn import (FastRCNNOutputLayers,
                                                     _log_classification_stats,
                                                     fast_rcnn_inference)
from torch.nn import functional as F

__all__ = ['CustomFastRCNNOutputLayers']


class CustomFastRCNNOutputLayers(FastRCNNOutputLayers):

    def __init__(self, cfg, input_shape: ShapeSpec, **kwargs):
        super().__init__(cfg, input_shape, **kwargs)

        self.cfg = cfg

    def losses(self, predictions, proposals):
        """enable advanced loss."""
        scores, proposal_deltas = predictions
        gt_classes = (
            cat([p.gt_classes for p in proposals], dim=0)
            if len(proposals) else torch.empty(0))
        # num_classes = self.num_classes
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
            self.box_reg_loss(proposal_boxes, gt_boxes, proposal_deltas,
                              gt_classes)
        }

    def sigmoid_cross_entropy_loss(self, pred_class_logits, gt_classes):
        if pred_class_logits.numel() == 0:
            return pred_class_logits.new_zeros(
                [1])[0]  # This is more robust than .sum() * 0.

        B = pred_class_logits.shape[0]
        C = pred_class_logits.shape[1] - 1

        target = pred_class_logits.new_zeros(B, C + 1)
        target[range(len(gt_classes)), gt_classes] = 1  # B x (C + 1)
        target = target[:, :C]  # B x C

        weight = 1

        cls_loss = F.binary_cross_entropy_with_logits(
            pred_class_logits[:, :-1], target, reduction='none')  # B x C
        loss = torch.sum(cls_loss * weight) / B
        return loss

    def softmax_cross_entropy_loss(self, pred_class_logits, gt_classes):
        """change _no_instance handling."""
        if pred_class_logits.numel() == 0:
            return pred_class_logits.new_zeros([1])[0]

        loss = F.cross_entropy(pred_class_logits, gt_classes, reduction='mean')
        return loss

    def inference(self, predictions, proposals):
        """enable use proposal boxes."""
        boxes = self.predict_boxes(predictions, proposals)
        scores = self.predict_probs(predictions, proposals)
        if self.cfg.MODEL.ROI_BOX_HEAD.MULT_PROPOSAL_SCORE:
            proposal_scores = [p.get('objectness_logits') for p in proposals]
            scores = [(s * ps[:, None])**0.5
                      for s, ps in zip(scores, proposal_scores)]
        image_shapes = [x.image_size for x in proposals]
        return fast_rcnn_inference(
            boxes,
            scores,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image,
        )

    def predict_probs(self, predictions, proposals):
        """support sigmoid."""
        scores, _ = predictions
        num_inst_per_image = [len(p) for p in proposals]
        probs = F.softmax(scores, dim=-1)
        return probs.split(num_inst_per_image, dim=0)
