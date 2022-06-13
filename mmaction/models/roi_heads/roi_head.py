# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union

import torch
from mmengine.config import ConfigDict
from mmengine.data import InstanceData
from torch import Tensor


from mmaction.core import ActionDataSample
from mmaction.registry import MODELS

try:
    from mmdet.models.roi_heads import StandardRoIHead
    from mmdet.core import bbox2roi, SamplingResult
    mmdet_imported = True
except (ImportError, ModuleNotFoundError):
    mmdet_imported = False

if mmdet_imported:

    @MODELS.register_module()
    class AVARoIHead(StandardRoIHead):

        def forward_train(self, x: Tuple[Tensor],
                          rpn_results_list: List[InstanceData],
                          batch_data_samples: List[ActionDataSample],
                          **kwargs) -> dict:
            """Forward function during training.

            Args:
                x (tuple[Tensor]): List of multi-level img features.
                rpn_results_list (list[:obj:`InstanceData`]): List of region
                    proposals.
                batch_data_samples (list[:obj:`DetDataSample`]): The batch
                    data samples. It usually includes information such
                    as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

            Returns:
                dict[str, Tensor]: A dictionary of loss components
            """
            assert len(rpn_results_list) == len(batch_data_samples)
            batch_gt_instances = []
            batch_gt_instances_ignore = []
            for data_sample in batch_data_samples:
                batch_gt_instances.append(data_sample.gt_instances)
                if 'ignored_instances' in data_sample:
                    batch_gt_instances_ignore.append(data_sample.ignored_instances)
                else:
                    batch_gt_instances_ignore.append(None)

            # assign gts and sample proposals
            num_imgs = len(batch_data_samples)
            sampling_results = []
            for i in range(num_imgs):
                # rename rpn_results.bboxes to rpn_results.priors
                rpn_results = rpn_results_list[i]
                rpn_results.priors = rpn_results.pop('bboxes')

                assign_result = self.bbox_assigner.assign(
                    rpn_results, batch_gt_instances[i],
                    batch_gt_instances_ignore[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    rpn_results,
                    batch_gt_instances[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

            # LFB needs meta_info: 'img_key'
            batch_size = len(batch_data_samples)
            batch_img_metas = []
            for batch_index in range(batch_size):
                metainfo = batch_data_samples[batch_index].metainfo
                batch_img_metas.append(metainfo)

            losses = dict()
            # bbox head forward and loss
            if self.with_bbox:
                bbox_results = self._bbox_forward_train(x, sampling_results,
                                                        batch_gt_instances,
                                                        batch_img_metas)
                losses.update(bbox_results['loss_bbox'])

            return losses

        def _bbox_forward(self, x, rois, batch_img_metas):
            """Defines the computation performed to get bbox predictions.

            Args:
                x (torch.Tensor): The input tensor.
                rois (torch.Tensor): The regions of interest.
                batch_img_metas (list): The meta info of images

            Returns:
                dict: bbox predictions with features and classification scores.
            """
            bbox_feat, global_feat = self.bbox_roi_extractor(x, rois)

            if self.with_shared_head:
                bbox_feat = self.shared_head(
                    bbox_feat,
                    feat=global_feat,
                    rois=rois,
                    img_metas=batch_img_metas)

            cls_score, bbox_pred = self.bbox_head(bbox_feat)

            bbox_results = dict(
                cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feat)
            return bbox_results

        def _bbox_forward_train(self, x: Tuple[Tensor],
                                sampling_results: List[SamplingResult],
                                batch_gt_instances: List[InstanceData],
                                batch_img_metas: List[dict]) -> dict:
            """Run forward function and calculate loss for box head in
            training."""
            rois = bbox2roi([res.bboxes for res in sampling_results])
            bbox_results = self._bbox_forward(x, rois, batch_img_metas)

            bbox_targets = self.bbox_head.get_targets(sampling_results,
                                                      self.train_cfg)
            loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                            bbox_results['bbox_pred'], rois,
                                            *bbox_targets)

            bbox_results.update(loss_bbox=loss_bbox)
            return bbox_results

        def simple_test(self,
                        x: Tuple[Tensor],
                        rpn_results_list: List[InstanceData],
                        batch_img_metas: List[dict],
                        rescale: bool = False,
                        **kwargs):
            """Defines the computation performed for simple testing."""
            assert self.with_bbox, 'Bbox head must be implemented.'

            if isinstance(x, tuple):
                x_shape = x[0].shape
            else:
                x_shape = x.shape

            assert x_shape[0] == 1, 'only accept 1 sample at test mode'
            assert x_shape[0] == len(batch_img_metas) == len(rpn_results_list)

            result_list = self.simple_test_bboxes(
                x,
                batch_img_metas,
                rpn_results_list,
                self.test_cfg,
                rescale=rescale)

            return result_list

        def simple_test_bboxes(self,
                               x: Tuple[Tensor],
                               batch_img_metas: List[dict],
                               rpn_results_list: List[InstanceData],
                               rcnn_test_cfg: ConfigDict,
                               rescale: bool = False) -> List[InstanceData]:
            """Test only det bboxes without augmentation."""
            proposals = [res.bboxes for res in rpn_results_list]
            rois = bbox2roi(proposals)
            bbox_results = self._bbox_forward(x, rois, batch_img_metas)
            cls_scores = bbox_results['cls_score']
            bbox_preds = bbox_results['bbox_pred']
            num_proposals_per_img = tuple(len(p) for p in proposals)
            rois = rois.split(num_proposals_per_img, 0)
            cls_scores = cls_scores.split(num_proposals_per_img, 0)

            if bbox_preds is None:
                bbox_preds = (None,) * len(proposals)

            result_list = self.bbox_head.get_results(
                rois=rois,
                cls_scores=cls_scores,
                bbox_preds=bbox_preds,  # List[None]
                batch_img_metas=batch_img_metas,
                rcnn_test_cfg=rcnn_test_cfg,
                rescale=rescale)

            return result_list
else:
    # Just define an empty class, so that __init__ can import it.
    class AVARoIHead:

        def __init__(self, *args, **kwargs):
            raise ImportError(
                'Failed to import `bbox2roi` from `mmdet.core.bbox`, '
                'or failed to import `MODELS` from `mmdet.registry`, '
                'or failed to import `StandardRoIHead` from '
                '`mmdet.models.roi_heads`. You will be unable to use '
                '`AVARoIHead`. ')
