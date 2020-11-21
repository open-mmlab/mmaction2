import numpy as np
import torch
import torch.nn as nn

from mmaction.core.bbox import (bbox2result, bbox2roi, build_assigner,
                                build_sampler)
from .. import builder
from ..registry import DETECTORS
from .base import BaseDetector


@DETECTORS.register_module
class FastRCNN(BaseDetector):

    def __init__(self,
                 backbone,
                 shared_head=None,
                 bbox_roi_extractor=None,
                 dropout_ratio=0,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):

        super(FastRCNN, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        self.shared_head = None
        self.bbox_roi_extractor = None
        self.bbox_head = None

        # shared_head is a res layer.
        if shared_head is not None:
            self.shared_head = builder.build_backbone(shared_head)

        if bbox_head is not None:
            self.bbox_roi_extractor = builder.build_roi_extractor(
                bbox_roi_extractor)
            self.bbox_head = builder.build_head(bbox_head)

        if dropout_ratio > 0:
            self.dropout = nn.Dropout(p=dropout_ratio)
        else:
            self.dropout = None

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights()

    def init_weights(self):
        super(FastRCNN, self).init_weights()
        self.backbone.init_weights()

        if self.shared_head:
            self.shared_head.init_weights()

        if self.bbox_head:
            self.bbox_roi_extractor.init_weights()
            self.bbox_head.init_weights()

    def forward_train(self, imgs, proposals, img_meta, entity_boxes, labels,
                      **kwargs):
        # Rename
        gt_bboxes = entity_boxes
        gt_labels = labels

        # Only 1 clip for each sample is permitted, due to the characteristic
        # of AVA
        assert imgs.shape[1] == 1
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])

        # proposals & gt_bboxes are padded, while scores & entity_ids are not.
        proposal_scores = [x['scores'] for x in img_meta]
        entity_ids = [x['entity_ids'] for x in img_meta]

        # Now we get the feature: it's N x C x T x H x W
        x = self.extract_feat(imgs)

        losses = dict()

        proposal_list = []
        for proposal, score in zip(proposals, proposal_scores):
            # which means to keep at least one proposal bbox
            num_proposal = len(score)
            proposal = proposal[:num_proposal]

            score_select_inds = score >= min(
                self.train_cfg.person_det_score_thr, max(score))
            # I think 25-pixel is a reasonable threshold
            area_select_inds = (proposal[:, 2] - proposal[:, 0]) * (
                proposal[:, 3] - proposal[:, 1]) > 25
            score_select_inds = torch.tensor(score_select_inds).to(
                area_select_inds.device)
            select_inds = score_select_inds & area_select_inds
            proposal_list.append(proposal[select_inds])

        gt_bbox_list = []
        gt_label_list = []
        for gt_bbox, gt_label, entity_id in zip(gt_bboxes, gt_labels,
                                                entity_ids):
            num_gt = len(entity_id)
            gt_bbox = gt_bbox[:num_gt]
            gt_label = gt_label[:num_gt]

            area_select_inds = (gt_bbox[:, 2] - gt_bbox[:, 0]) * (
                gt_bbox[:, 3] - gt_bbox[:, 1]) > 25
            gt_bbox_list.append(gt_bbox[area_select_inds])
            gt_label_list.append(gt_label[area_select_inds])

        # assign gts and proposals, note that theoretically, proposal_list or
        # gt_bbox_list can be 0, so we need to be careful

        if self.bbox_head:
            bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
            bbox_sampler = build_sampler(self.train_cfg.rcnn.sampler)
            num_imgs = imgs.size(0)

            sampling_results = []
            img_inds = []
            for i in range(num_imgs):
                if not (len(proposal_list[i]) and len(gt_bbox_list[i])):
                    img_inds.append(False)
                    continue
                assign_result = bbox_assigner.assign(proposal_list[i],
                                                     gt_bbox_list[i],
                                                     gt_label_list[i])
                sampling_result = bbox_sampler.sample(assign_result,
                                                      proposal_list[i],
                                                      gt_bboxes[i],
                                                      gt_labels[i])
                sampling_results.append(sampling_result)
                img_inds.append(True)

            img_inds = torch.tensor(img_inds)
            # Extract feature with inds
            x = (x[0][img_inds], )

        # bbox head forward and loss
        if self.bbox_head:
            rois = bbox2roi([res.bboxes for res in sampling_results])
            # Now we can only deal with one feature map
            bbox_feats = self.bbox_roi_extractor(x[0], rois)

            if self.shared_head:
                bbox_feats = self.shared_head(bbox_feats)

            if self.dropout is not None:
                bbox_feats = self.dropout(bbox_feats)

            cls_score = self.bbox_head(bbox_feats)

            bbox_targets = self.bbox_head.get_target(sampling_results,
                                                     gt_labels,
                                                     self.train_cfg.rcnn)
            loss_bbox = self.bbox_head.loss(cls_score, *bbox_targets)
            losses.update(loss_bbox)

        return losses

    def forward_test(self, imgs, proposals, img_meta, **kwargs):
        """Test without augmentation."""
        assert self.bbox_head, 'Bbox head must be implemented.'

        # batch_size 1 & 1 clip
        assert imgs.shape[:2] == (1, 1)
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])

        x = self.extract_feat(imgs)
        assert len(img_meta) == 1

        proposal_scores = [x['scores'] for x in img_meta]

        proposal_list = []
        for proposal, score in zip(proposals, proposal_scores):
            num_proposal = len(score)
            proposal = proposal[:num_proposal]

            score_select_inds = score >= min(
                self.test_cfg.person_det_score_thr, max(score))
            area_select_inds = (proposal[:, 2] - proposal[:, 0]) * (
                proposal[:, 3] - proposal[:, 1]) > 25
            score_select_inds = torch.tensor(score_select_inds).to(
                area_select_inds.device)
            select_inds = score_select_inds & area_select_inds
            proposal_list.append(proposal[select_inds])

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_meta, proposal_list, self.test_cfg.rcnn)

        bbox_results = bbox2result(
            det_bboxes,
            det_labels,
            self.bbox_head.num_classes,
            thr=self.test_cfg.rcnn.action_thr)

        # Since only 1 sample here
        return [bbox_results]

    def simple_test_bboxes(self, x, img_meta, proposals, rcnn_test_cfg):
        """Test only det bboxes without augmentation."""
        # rois have batch_ind, do not have scores
        rois = bbox2roi(proposals)
        roi_feats = self.bbox_roi_extractor(x[0], rois)

        if self.shared_head:
            roi_feats = self.shared_head(roi_feats)

        cls_score = self.bbox_head(roi_feats)

        # img_shape is required, crop_quadruple and flip are optional
        img_shape = img_meta[0]['img_shape']

        crop_quadruple = np.array([0, 0, 1, 1])
        flip = False

        if 'crop_quadruple' in img_meta[0]:
            crop_quadruple = img_meta[0]['crop_quadruple']

        # If flip used, we should first flip the proposal box
        if 'flip' in img_meta[0]:
            flip = img_meta[0]['flip']

        # The returned det_bboxes are normalized to [0, 1]
        det_bboxes, det_labels = self.bbox_head.get_det_bboxes(
            rois,
            cls_score,
            img_shape,
            flip=flip,
            crop_quadruple=crop_quadruple,
            cfg=rcnn_test_cfg)

        return det_bboxes, det_labels
