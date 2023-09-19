import logging
import math
from typing import List, Tuple

import torch
from detectron2.config import configurable
from detectron2.layers import batched_nms
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads.cascade_rcnn import (CascadeROIHeads,
                                                        _ScaleGradient)
from detectron2.modeling.roi_heads.roi_heads import ROI_HEADS_REGISTRY
from detectron2.structures import Boxes, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from models.grit_src.grit.data.custom_dataset_mapper import ObjDescription
from transformers import BertTokenizer

from ..soft_nms import batched_soft_nms
from ..text.load_text_token import LoadTextTokens
from ..text.text_decoder import (AutoRegressiveBeamSearch, GRiTTextDecoder,
                                 TransformerDecoderTextualHead)
from .grit_fast_rcnn import GRiTFastRCNNOutputLayers

logger = logging.getLogger(__name__)


@ROI_HEADS_REGISTRY.register()
class GRiTROIHeadsAndTextDecoder(CascadeROIHeads):

    @configurable
    def __init__(
        self,
        *,
        text_decoder_transformer,
        train_task: list,
        test_task: str,
        mult_proposal_score: bool = False,
        mask_weight: float = 1.0,
        object_feat_pooler=None,
        soft_nms_enabled=False,
        beam_size=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mult_proposal_score = mult_proposal_score
        self.mask_weight = mask_weight
        self.object_feat_pooler = object_feat_pooler
        self.soft_nms_enabled = soft_nms_enabled
        self.test_task = test_task
        self.beam_size = beam_size

        tokenizer = BertTokenizer.from_pretrained(
            '/mnt/data.coronaryct.1/ZhuYichen/Ask-Anything/model/bert-base'
            '-uncased',
            local_files_only=True)
        self.tokenizer = tokenizer

        assert test_task in train_task, 'GRiT has not been trained on {} ' \
                                        'task, ' \
                                        'please verify the task name or ' \
                                        'train a new ' \
                                        'GRiT on {} task'.format(test_task,
                                                                 test_task)
        task_begin_tokens = {}
        for i, task in enumerate(train_task):
            if i == 0:
                task_begin_tokens[task] = tokenizer.cls_token_id
            else:
                task_begin_tokens[task] = 103 + i
        self.task_begin_tokens = task_begin_tokens

        beamsearch_decode = AutoRegressiveBeamSearch(
            end_token_id=tokenizer.sep_token_id,
            max_steps=40,
            beam_size=beam_size,
            objectdet=test_task == 'ObjectDet',
            per_node_beam_size=1,
        )
        self.text_decoder = GRiTTextDecoder(
            text_decoder_transformer,
            beamsearch_decode=beamsearch_decode,
            begin_token_id=task_begin_tokens[test_task],
            loss_type='smooth',
            tokenizer=tokenizer,
        )
        self.get_target_text_tokens = LoadTextTokens(
            tokenizer, max_text_len=40, padding='do_not_pad')

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        text_decoder_transformer = TransformerDecoderTextualHead(
            object_feature_size=cfg.MODEL.FPN.OUT_CHANNELS,
            vocab_size=cfg.TEXT_DECODER.VOCAB_SIZE,
            hidden_size=cfg.TEXT_DECODER.HIDDEN_SIZE,
            num_layers=cfg.TEXT_DECODER.NUM_LAYERS,
            attention_heads=cfg.TEXT_DECODER.ATTENTION_HEADS,
            feedforward_size=cfg.TEXT_DECODER.FEEDFORWARD_SIZE,
            mask_future_positions=True,
            padding_idx=0,
            decoder_type='bert_en',
            use_act_checkpoint=cfg.USE_ACT_CHECKPOINT,
        )
        ret.update({
            'text_decoder_transformer': text_decoder_transformer,
            'train_task': cfg.MODEL.TRAIN_TASK,
            'test_task': cfg.MODEL.TEST_TASK,
            'mult_proposal_score': cfg.MODEL.ROI_BOX_HEAD.MULT_PROPOSAL_SCORE,
            'mask_weight': cfg.MODEL.ROI_HEADS.MASK_WEIGHT,
            'soft_nms_enabled': cfg.MODEL.ROI_HEADS.SOFT_NMS_ENABLED,
            'beam_size': cfg.MODEL.BEAM_SIZE,
        })
        return ret

    @classmethod
    def _init_box_head(self, cfg, input_shape):
        ret = super()._init_box_head(cfg, input_shape)
        del ret['box_predictors']
        cascade_bbox_reg_weights = \
            cfg.MODEL.ROI_BOX_CASCADE_HEAD.BBOX_REG_WEIGHTS
        box_predictors = []
        for box_head, bbox_reg_weights in zip(ret['box_heads'],
                                              cascade_bbox_reg_weights):
            box_predictors.append(
                GRiTFastRCNNOutputLayers(
                    cfg,
                    box_head.output_shape,
                    box2box_transform=Box2BoxTransform(
                        weights=bbox_reg_weights)))
        ret['box_predictors'] = box_predictors

        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        object_feat_pooler = ROIPooler(
            output_size=cfg.MODEL.ROI_HEADS.OBJECT_FEAT_POOLER_RES,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        ret['object_feat_pooler'] = object_feat_pooler
        return ret

    def check_if_all_background(self, proposals, targets, stage):
        all_background = True
        for proposals_per_image in proposals:
            if not (proposals_per_image.gt_classes == self.num_classes).all():
                all_background = False

        if all_background:
            logger.info(
                'all proposals are background at stage {}'.format(stage))
            proposals[0].proposal_boxes.tensor[0, :] = \
                targets[0].gt_boxes.tensor[0, :]
            proposals[0].gt_boxes.tensor[0, :] = targets[0].gt_boxes.tensor[
                0, :]
            proposals[0].objectness_logits[0] = math.log(
                (1.0 - 1e-10) / (1 - (1.0 - 1e-10)))
            proposals[0].gt_classes[0] = targets[0].gt_classes[0]
            proposals[0].gt_object_descriptions.data[0] = targets[
                0].gt_object_descriptions.data[0]
            if 'foreground' in proposals[0].get_fields().keys():
                proposals[0].foreground[0] = 1
        return proposals

    def _forward_box(self,
                     features,
                     proposals,
                     targets=None,
                     task='ObjectDet'):
        if self.training:
            proposals = self.check_if_all_background(proposals, targets, 0)
        if (not self.training) and self.mult_proposal_score:
            if len(proposals) > 0 and proposals[0].has('scores'):
                proposal_scores = [p.get('scores') for p in proposals]
            else:
                proposal_scores = [
                    p.get('objectness_logits') for p in proposals
                ]

        features = [features[f] for f in self.box_in_features]
        head_outputs = []
        prev_pred_boxes = None
        image_sizes = [x.image_size for x in proposals]

        for k in range(self.num_cascade_stages):
            if k > 0:
                proposals = self._create_proposals_from_boxes(
                    prev_pred_boxes,
                    image_sizes,
                    logits=[p.objectness_logits for p in proposals])
                if self.training:
                    proposals = self._match_and_label_boxes_GRiT(
                        proposals, k, targets)
                    proposals = self.check_if_all_background(
                        proposals, targets, k)
            predictions = self._run_stage(features, proposals, k)
            prev_pred_boxes = self.box_predictor[k].predict_boxes(
                (predictions[0], predictions[1]), proposals)
            head_outputs.append(
                (self.box_predictor[k], predictions, proposals))

        if self.training:
            object_features = self.object_feat_pooler(
                features, [x.proposal_boxes for x in proposals])
            object_features = _ScaleGradient.apply(
                object_features, 1.0 / self.num_cascade_stages)
            foreground = torch.cat([x.foreground for x in proposals])
            object_features = object_features[foreground > 0]

            object_descriptions = []
            for x in proposals:
                object_descriptions += x.gt_object_descriptions[
                    x.foreground > 0].data
            object_descriptions = ObjDescription(object_descriptions)
            object_descriptions = object_descriptions.data

            if len(object_descriptions) > 0:
                begin_token = self.task_begin_tokens[task]
                text_decoder_inputs = self.get_target_text_tokens(
                    object_descriptions, object_features, begin_token)
                object_features = object_features.view(
                    object_features.shape[0], object_features.shape[1],
                    -1).permute(0, 2, 1).contiguous()
                text_decoder_inputs.update(
                    {'object_features': object_features})
                text_decoder_loss = self.text_decoder(text_decoder_inputs)
            else:
                text_decoder_loss = head_outputs[0][1][0].new_zeros([1])[0]

            losses = {}
            storage = get_event_storage()
            # RoI Head losses (For the proposal generator loss, please find
            # it in grit.py)
            for stage, (predictor, predictions,
                        proposals) in enumerate(head_outputs):
                with storage.name_scope('stage{}'.format(stage)):
                    stage_losses = predictor.losses(
                        (predictions[0], predictions[1]), proposals)
                losses.update({
                    k + '_stage{}'.format(stage): v
                    for k, v in stage_losses.items()
                })
            # Text Decoder loss
            losses.update({'text_decoder_loss': text_decoder_loss})
            return losses
        else:
            scores_per_stage = [
                h[0].predict_probs(h[1], h[2]) for h in head_outputs
            ]
            logits_per_stage = [(h[1][0], ) for h in head_outputs]
            scores = [
                sum(list(scores_per_image)) * (1.0 / self.num_cascade_stages)
                for scores_per_image in zip(*scores_per_stage)
            ]
            logits = [
                sum(list(logits_per_image)) * (1.0 / self.num_cascade_stages)
                for logits_per_image in zip(*logits_per_stage)
            ]
            if self.mult_proposal_score:
                scores = [(s * ps[:, None])**0.5
                          for s, ps in zip(scores, proposal_scores)]
            predictor, predictions, proposals = head_outputs[-1]
            boxes = predictor.predict_boxes((predictions[0], predictions[1]),
                                            proposals)
            assert len(boxes) == 1
            pred_instances, _ = self.fast_rcnn_inference_GRiT(
                boxes,
                scores,
                logits,
                image_sizes,
                predictor.test_score_thresh,
                predictor.test_nms_thresh,
                predictor.test_topk_per_image,
                self.soft_nms_enabled,
            )

            assert len(pred_instances) == 1, 'Only support one image'
            for i, pred_instance in enumerate(pred_instances):
                if len(pred_instance.pred_boxes) > 0:
                    object_features = self.object_feat_pooler(
                        features, [pred_instance.pred_boxes])
                    object_features = object_features.view(
                        object_features.shape[0], object_features.shape[1],
                        -1).permute(0, 2, 1).contiguous()
                    text_decoder_output = self.text_decoder(
                        {'object_features': object_features})
                    if self.beam_size > 1 and self.test_task == 'ObjectDet':
                        pred_boxes = []
                        pred_scores = []
                        pred_classes = []
                        pred_object_descriptions = []

                        for beam_id in range(self.beam_size):
                            pred_boxes.append(pred_instance.pred_boxes.tensor)
                            # object score = sqrt(objectness score x
                            # description score)
                            pred_scores.append(
                                (pred_instance.scores *
                                 torch.exp(text_decoder_output['logprobs'])
                                 [:, beam_id])**0.5)
                            pred_classes.append(pred_instance.pred_classes)
                            for prediction in \
                                    text_decoder_output[
                                        'predictions'][:, beam_id, :]:
                                # convert text tokens to words
                                description = self.tokenizer.decode(
                                    prediction.tolist()[1:],
                                    skip_special_tokens=True)
                                pred_object_descriptions.append(description)

                        merged_instances = Instances(image_sizes[0])
                        if torch.cat(
                                pred_scores, dim=0
                        ).shape[0] <= predictor.test_topk_per_image:
                            merged_instances.scores = torch.cat(
                                pred_scores, dim=0)
                            merged_instances.pred_boxes = Boxes(
                                torch.cat(pred_boxes, dim=0))
                            merged_instances.pred_classes = torch.cat(
                                pred_classes, dim=0)
                            merged_instances.pred_object_descriptions = \
                                ObjDescription(pred_object_descriptions)
                        else:
                            pred_scores, top_idx = torch.topk(
                                torch.cat(pred_scores, dim=0),
                                predictor.test_topk_per_image)
                            merged_instances.scores = pred_scores
                            merged_instances.pred_boxes = Boxes(
                                torch.cat(pred_boxes, dim=0)[top_idx, :])
                            merged_instances.pred_classes = torch.cat(
                                pred_classes, dim=0)[top_idx]
                            merged_instances.pred_object_descriptions = \
                                ObjDescription(
                                    ObjDescription(pred_object_descriptions)[
                                        top_idx].data)

                        pred_instances[i] = merged_instances
                    else:
                        # object score = sqrt(objectness score x description
                        # score)
                        pred_instance.scores = \
                            (pred_instance.scores * torch.exp(
                                text_decoder_output['logprobs'])) ** 0.5

                        pred_object_descriptions = []
                        for prediction in text_decoder_output['predictions']:
                            # convert text tokens to words
                            description = self.tokenizer.decode(
                                prediction.tolist()[1:],
                                skip_special_tokens=True)
                            pred_object_descriptions.append(description)
                        pred_instance.pred_object_descriptions = \
                            ObjDescription(pred_object_descriptions)
                else:
                    pred_instance.pred_object_descriptions = ObjDescription([])

            return pred_instances

    def forward(self,
                features,
                proposals,
                targets=None,
                targets_task='ObjectDet'):
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)

            losses = self._forward_box(
                features, proposals, targets, task=targets_task)
            if targets[0].has('gt_masks'):
                mask_losses = self._forward_mask(features, proposals)
                losses.update(
                    {k: v * self.mask_weight
                     for k, v in mask_losses.items()})
            else:
                losses.update(
                    self._get_empty_mask_loss(
                        device=proposals[0].objectness_logits.device))
            return proposals, losses
        else:
            pred_instances = self._forward_box(
                features, proposals, task=self.test_task)
            pred_instances = self.forward_with_given_boxes(
                features, pred_instances)
            return pred_instances, {}

    @torch.no_grad()
    def _match_and_label_boxes_GRiT(self, proposals, stage, targets):
        """Add  "gt_object_description" and "foreground" to detectron2's
        _match_and_label_boxes."""
        num_fg_samples, num_bg_samples = [], []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes)
            # proposal_labels are 0 or 1
            matched_idxs, proposal_labels = self.proposal_matchers[stage](
                match_quality_matrix)
            if len(targets_per_image) > 0:
                gt_classes = targets_per_image.gt_classes[matched_idxs]
                # Label unmatched proposals (0 label from matcher) as
                # background (label=num_classes)
                gt_classes[proposal_labels == 0] = self.num_classes
                foreground = torch.ones_like(gt_classes)
                foreground[proposal_labels == 0] = 0
                gt_boxes = targets_per_image.gt_boxes[matched_idxs]
                gt_object_descriptions = \
                    targets_per_image.gt_object_descriptions[
                        matched_idxs]
            else:
                gt_classes = torch.zeros_like(matched_idxs) + self.num_classes
                foreground = torch.zeros_like(gt_classes)
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros(
                        (len(proposals_per_image), 4)))
                gt_object_descriptions = ObjDescription(
                    ['None' for i in range(len(proposals_per_image))])
            proposals_per_image.gt_classes = gt_classes
            proposals_per_image.gt_boxes = gt_boxes
            proposals_per_image.gt_object_descriptions = gt_object_descriptions
            proposals_per_image.foreground = foreground

            num_fg_samples.append((proposal_labels == 1).sum().item())
            num_bg_samples.append(proposal_labels.numel() - num_fg_samples[-1])

        # Log the number of fg/bg samples in each stage
        storage = get_event_storage()
        storage.put_scalar(
            'stage{}/roi_head/num_fg_samples'.format(stage),
            sum(num_fg_samples) / len(num_fg_samples),
        )
        storage.put_scalar(
            'stage{}/roi_head/num_bg_samples'.format(stage),
            sum(num_bg_samples) / len(num_bg_samples),
        )
        return proposals

    def fast_rcnn_inference_GRiT(
        self,
        boxes: List[torch.Tensor],
        scores: List[torch.Tensor],
        logits: List[torch.Tensor],
        image_shapes: List[Tuple[int, int]],
        score_thresh: float,
        nms_thresh: float,
        topk_per_image: int,
        soft_nms_enabled: bool,
    ):
        result_per_image = [
            self.fast_rcnn_inference_single_image_GRiT(boxes_per_image,
                                                       scores_per_image,
                                                       logits_per_image,
                                                       image_shape,
                                                       score_thresh,
                                                       nms_thresh,
                                                       topk_per_image,
                                                       soft_nms_enabled)
            for scores_per_image, boxes_per_image, image_shape,
            logits_per_image in zip(scores, boxes, image_shapes, logits)
        ]
        return [x[0]
                for x in result_per_image], [x[1] for x in result_per_image]

    def fast_rcnn_inference_single_image_GRiT(
        self,
        boxes,
        scores,
        logits,
        image_shape: Tuple[int, int],
        score_thresh: float,
        nms_thresh: float,
        topk_per_image: int,
        soft_nms_enabled,
    ):
        """Add soft NMS to detectron2's fast_rcnn_inference_single_image."""
        valid_mask = torch.isfinite(boxes).all(
            dim=1) & torch.isfinite(scores).all(dim=1)
        if not valid_mask.all():
            boxes = boxes[valid_mask]
            scores = scores[valid_mask]
            logits = logits[valid_mask]

        scores = scores[:, :-1]
        logits = logits[:, :-1]
        num_bbox_reg_classes = boxes.shape[1] // 4
        # Convert to Boxes to use the `clip` function ...
        boxes = Boxes(boxes.reshape(-1, 4))
        boxes.clip(image_shape)
        boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

        # 1. Filter results based on detection scores. It can make NMS more
        # efficient by filtering out low-confidence detections.
        filter_mask = scores > score_thresh  # R x K
        # R' x 2. First column contains indices of the R predictions;
        # Second column contains indices of classes.
        filter_inds = filter_mask.nonzero()
        if num_bbox_reg_classes == 1:
            boxes = boxes[filter_inds[:, 0], 0]
        else:
            boxes = boxes[filter_mask]
        scores = scores[filter_mask]
        logits = logits[filter_mask]

        # 2. Apply NMS for each class independently.
        if not soft_nms_enabled:
            keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
        else:
            keep, soft_nms_scores = batched_soft_nms(
                boxes,
                scores,
                filter_inds[:, 1],
                'linear',
                0.5,
                nms_thresh,
                0.001,
            )
            scores[keep] = soft_nms_scores
        if topk_per_image >= 0:
            keep = keep[:topk_per_image]
        boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[
            keep]
        logits = logits[keep]

        result = Instances(image_shape)
        result.pred_boxes = Boxes(boxes)
        result.scores = scores
        result.pred_classes = filter_inds[:, 1]
        result.logits = logits
        return result, filter_inds[:, 0]

    def _get_empty_mask_loss(self, device):
        if self.mask_on:
            return {
                'loss_mask':
                torch.zeros((1, ), device=device, dtype=torch.float32)[0]
            }
        else:
            return {}

    def _create_proposals_from_boxes(self, boxes, image_sizes, logits):
        boxes = [Boxes(b.detach()) for b in boxes]
        proposals = []
        for boxes_per_image, image_size, logit in zip(boxes, image_sizes,
                                                      logits):
            boxes_per_image.clip(image_size)
            if self.training:
                inds = boxes_per_image.nonempty()
                boxes_per_image = boxes_per_image[inds]
                logit = logit[inds]
            prop = Instances(image_size)
            prop.proposal_boxes = boxes_per_image
            prop.objectness_logits = logit
            proposals.append(prop)
        return proposals

    def _run_stage(self, features, proposals, stage):
        pool_boxes = [x.proposal_boxes for x in proposals]
        box_features = self.box_pooler(features, pool_boxes)
        box_features = _ScaleGradient.apply(box_features,
                                            1.0 / self.num_cascade_stages)
        box_features = self.box_head[stage](box_features)
        return self.box_predictor[stage](box_features)
