# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.roi_heads.cascade_rcnn import CascadeROIHeads
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference
from detectron2.modeling.roi_heads.roi_heads import (ROI_HEADS_REGISTRY,
                                                     StandardROIHeads)
from detectron2.utils.events import get_event_storage

from .custom_fast_rcnn import CustomFastRCNNOutputLayers


@ROI_HEADS_REGISTRY.register()
class CustomROIHeads(StandardROIHeads):

    @classmethod
    def _init_box_head(self, cfg, input_shape):
        ret = super()._init_box_head(cfg, input_shape)
        del ret['box_predictor']
        ret['box_predictor'] = CustomFastRCNNOutputLayers(
            cfg, ret['box_head'].output_shape)
        self.debug = cfg.DEBUG
        if self.debug:
            self.debug_show_name = cfg.DEBUG_SHOW_NAME
            self.save_debug = cfg.SAVE_DEBUG
            self.vis_thresh = cfg.VIS_THRESH
            self.pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(
                torch.device(cfg.MODEL.DEVICE)).view(3, 1, 1)
            self.pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(
                torch.device(cfg.MODEL.DEVICE)).view(3, 1, 1)
        return ret

    def forward(self, images, features, proposals, targets=None):
        """enable debug."""
        if not self.debug:
            del images
        if self.training:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        if self.training:
            losses = self._forward_box(features, proposals)
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            pred_instances = self.forward_with_given_boxes(
                features, pred_instances)
            if self.debug:
                from ..debug import debug_second_stage
                debug_second_stage([self.denormalizer(images[0].clone())],
                                   pred_instances,
                                   proposals=proposals,
                                   debug_show_name=self.debug_show_name)
            return pred_instances, {}

    def denormalizer(self, x):
        return x * self.pixel_std + self.pixel_mean


@ROI_HEADS_REGISTRY.register()
class CustomCascadeROIHeads(CascadeROIHeads):

    @classmethod
    def _init_box_head(self, cfg, input_shape):
        self.mult_proposal_score = cfg.MODEL.ROI_BOX_HEAD.MULT_PROPOSAL_SCORE
        ret = super()._init_box_head(cfg, input_shape)
        del ret['box_predictors']
        cascade_bbox_reg_weights = \
            cfg.MODEL.ROI_BOX_CASCADE_HEAD.BBOX_REG_WEIGHTS
        box_predictors = []
        for box_head, bbox_reg_weights in zip(ret['box_heads'],
                                              cascade_bbox_reg_weights):
            box_predictors.append(
                CustomFastRCNNOutputLayers(
                    cfg,
                    box_head.output_shape,
                    box2box_transform=Box2BoxTransform(
                        weights=bbox_reg_weights)))
        ret['box_predictors'] = box_predictors
        self.debug = cfg.DEBUG
        if self.debug:
            self.debug_show_name = cfg.DEBUG_SHOW_NAME
            self.save_debug = cfg.SAVE_DEBUG
            self.vis_thresh = cfg.VIS_THRESH
            self.pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(
                torch.device(cfg.MODEL.DEVICE)).view(3, 1, 1)
            self.pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(
                torch.device(cfg.MODEL.DEVICE)).view(3, 1, 1)
        return ret

    def _forward_box(self, features, proposals, targets=None):
        """Add mult proposal scores at testing."""
        if (not self.training) and self.mult_proposal_score:
            if len(proposals) > 0 and proposals[0].has('scores'):
                proposal_scores = [p.get('scores') for p in proposals]
            else:
                proposal_scores = [
                    p.get('objectness_logits') for p in proposals
                ]

        features = [features[f] for f in self.box_in_features]
        head_outputs = []  # (predictor, predictions, proposals)
        prev_pred_boxes = None
        image_sizes = [x.image_size for x in proposals]
        for k in range(self.num_cascade_stages):
            if k > 0:
                proposals = self._create_proposals_from_boxes(
                    prev_pred_boxes, image_sizes)
                if self.training:
                    proposals = self._match_and_label_boxes(
                        proposals, k, targets)
            predictions = self._run_stage(features, proposals, k)
            prev_pred_boxes = self.box_predictor[k].predict_boxes(
                predictions, proposals)
            head_outputs.append(
                (self.box_predictor[k], predictions, proposals))

        if self.training:
            losses = {}
            storage = get_event_storage()
            for stage, (predictor, predictions,
                        proposals) in enumerate(head_outputs):
                with storage.name_scope('stage{}'.format(stage)):
                    stage_losses = predictor.losses(predictions, proposals)
                losses.update({
                    k + '_stage{}'.format(stage): v
                    for k, v in stage_losses.items()
                })
            return losses
        else:
            # Each is a list[Tensor] of length #image. Each tensor is Ri x (
            # K+1)
            scores_per_stage = [
                h[0].predict_probs(h[1], h[2]) for h in head_outputs
            ]
            scores = [
                sum(list(scores_per_image)) * (1.0 / self.num_cascade_stages)
                for scores_per_image in zip(*scores_per_stage)
            ]

            if self.mult_proposal_score:
                scores = [(s * ps[:, None])**0.5
                          for s, ps in zip(scores, proposal_scores)]

            predictor, predictions, proposals = head_outputs[-1]
            boxes = predictor.predict_boxes(predictions, proposals)
            pred_instances, _ = fast_rcnn_inference(
                boxes,
                scores,
                image_sizes,
                predictor.test_score_thresh,
                predictor.test_nms_thresh,
                predictor.test_topk_per_image,
            )

            return pred_instances

    def forward(self, images, features, proposals, targets=None):
        """enable debug."""
        if not self.debug:
            del images
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)

        if self.training:
            losses = self._forward_box(features, proposals, targets)
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            # import pdb; pdb.set_trace()
            pred_instances = self._forward_box(features, proposals)
            pred_instances = self.forward_with_given_boxes(
                features, pred_instances)
            if self.debug:
                from ..debug import debug_second_stage
                debug_second_stage(
                    [self.denormalizer(x.clone()) for x in images],
                    pred_instances,
                    proposals=proposals,
                    save_debug=self.save_debug,
                    debug_show_name=self.debug_show_name,
                    vis_thresh=self.vis_thresh)
            return pred_instances, {}

    def denormalizer(self, x):
        return x * self.pixel_std + self.pixel_mean
