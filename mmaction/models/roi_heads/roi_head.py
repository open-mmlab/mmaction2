# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

from torch import Tensor

from mmaction.utils import ConfigType, InstanceList, SampleList

try:
    from mmdet.models.roi_heads import StandardRoIHead
    from mmdet.models.task_modules.samplers import SamplingResult
    from mmdet.registry import MODELS as MMDET_MODELS
    from mmdet.structures.bbox import bbox2roi
    mmdet_imported = True
except (ImportError, ModuleNotFoundError):
    from mmaction.utils import SamplingResult
    mmdet_imported = False

if mmdet_imported:

    @MMDET_MODELS.register_module()
    class AVARoIHead(StandardRoIHead):

        def loss(self, x: Union[Tensor,
                                Tuple[Tensor]], rpn_results_list: InstanceList,
                 data_samples: SampleList, **kwargs) -> dict:
            """Perform forward propagation and loss calculation of the
            detection roi on the features of the upstream network.

            Args:
                x (Tensor or Tuple[Tensor]): The image features extracted by
                    the upstream network.
                rpn_results_list (List[:obj:`InstanceData`]): List of region
                    proposals.
                data_samples (List[:obj:`ActionDataSample`]): The batch
                    data samples.

            Returns:
                Dict[str, Tensor]: A dictionary of loss components.
            """
            assert len(rpn_results_list) == len(data_samples)
            batch_gt_instances = []
            for data_sample in data_samples:
                batch_gt_instances.append(data_sample.gt_instances)

            # assign gts and sample proposals
            num_imgs = len(data_samples)
            sampling_results = []
            for i in range(num_imgs):
                # rename rpn_results.bboxes to rpn_results.priors
                rpn_results = rpn_results_list[i]
                rpn_results.priors = rpn_results.pop('bboxes')

                assign_result = self.bbox_assigner.assign(
                    rpn_results, batch_gt_instances[i], None)
                sampling_result = self.bbox_sampler.sample(
                    assign_result, rpn_results, batch_gt_instances[i])
                sampling_results.append(sampling_result)

            # LFB needs meta_info: 'img_key'
            batch_img_metas = [
                data_samples.metainfo for data_samples in data_samples
            ]

            losses = dict()
            # bbox head forward and loss
            bbox_results = self.bbox_loss(x, sampling_results, batch_img_metas)
            losses.update(bbox_results['loss_bbox'])

            return losses

        def _bbox_forward(self, x: Union[Tensor, Tuple[Tensor]], rois: Tensor,
                          batch_img_metas: List[dict], **kwargs) -> dict:
            """Box head forward function used in both training and testing.

            Args:
                x (Tensor or Tuple[Tensor]): The image features extracted by
                    the upstream network.
                rois (Tensor): RoIs with the shape (n, 5) where the first
                    column indicates batch id of each RoI.
                batch_img_metas (List[dict]): List of image information.

            Returns:
                 dict[str, Tensor]: Usually returns a dictionary with keys:

                    - `cls_score` (Tensor): Classification scores.
                    - `bbox_pred` (Tensor): Box energies / deltas.
                    - `bbox_feats` (Tensor): Extract bbox RoI features.
            """
            bbox_feats, global_feat = self.bbox_roi_extractor(x, rois)

            if self.with_shared_head:
                bbox_feats = self.shared_head(
                    bbox_feats,
                    feat=global_feat,
                    rois=rois,
                    img_metas=batch_img_metas)

            cls_score = self.bbox_head(bbox_feats)

            bbox_results = dict(cls_score=cls_score, bbox_feats=bbox_feats)
            return bbox_results

        def bbox_loss(self, x: Union[Tensor, Tuple[Tensor]],
                      sampling_results: List[SamplingResult],
                      batch_img_metas: List[dict], **kwargs) -> dict:
            """Perform forward propagation and loss calculation of the bbox
            head on the features of the upstream network.

            Args:
                x (Tensor or Tuple[Tensor]): The image features extracted by
                    the upstream network.
                sampling_results (List[SamplingResult]): Sampling results.
                batch_img_metas (List[dict]): List of image information.

            Returns:
                dict[str, Tensor]: Usually returns a dictionary with keys:

                    - `cls_score` (Tensor): Classification scores.
                    - `bbox_pred` (Tensor): Box energies / deltas.
                    - `bbox_feats` (Tensor): Extract bbox RoI features.
                    - `loss_bbox` (dict): A dictionary of bbox loss components.
            """
            rois = bbox2roi([res.priors for res in sampling_results])
            bbox_results = self._bbox_forward(x, rois, batch_img_metas)

            bbox_loss_and_target = self.bbox_head.loss_and_target(
                cls_score=bbox_results['cls_score'],
                rois=rois,
                sampling_results=sampling_results,
                rcnn_train_cfg=self.train_cfg)

            bbox_results.update(loss_bbox=bbox_loss_and_target['loss_bbox'])
            return bbox_results

        def predict(self, x: Union[Tensor, Tuple[Tensor]],
                    rpn_results_list: InstanceList, data_samples: SampleList,
                    **kwargs) -> InstanceList:
            """Perform forward propagation of the roi head and predict
            detection results on the features of the upstream network.

            Args:
                x (Tensor or Tuple[Tensor]): The image features extracted by
                    the upstream network.
                rpn_results_list (List[:obj:`InstanceData`]): list of region
                    proposals.
                data_samples (List[:obj:`ActionDataSample`]): The batch
                    data samples.

            Returns:
                List[obj:`InstanceData`]: Detection results of each image.
                Each item usually contains following keys.

                    - scores (Tensor): Classification scores, has a shape
                      (num_instance, )
                    - labels (Tensor): Labels of bboxes, has a shape
                      (num_instances, ).
            """
            assert self.with_bbox, 'Bbox head must be implemented.'
            batch_img_metas = [
                data_samples.metainfo for data_samples in data_samples
            ]
            if isinstance(x, tuple):
                x_shape = x[0].shape
            else:
                x_shape = x.shape

            assert x_shape[0] == 1, 'only accept 1 sample at test mode'
            assert x_shape[0] == len(batch_img_metas) == len(rpn_results_list)

            results_list = self.predict_bbox(
                x,
                batch_img_metas,
                rpn_results_list,
                rcnn_test_cfg=self.test_cfg)

            return results_list

        def predict_bbox(self, x: Tuple[Tensor], batch_img_metas: List[dict],
                         rpn_results_list: InstanceList,
                         rcnn_test_cfg: ConfigType) -> InstanceList:
            """Perform forward propagation of the bbox head and predict
            detection results on the features of the upstream network.

            Args:
                x (tuple[Tensor]): Feature maps of all scale level.
                batch_img_metas (list[dict]): List of image information.
                rpn_results_list (list[:obj:`InstanceData`]): List of region
                    proposals.
                rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.

            Returns:
                list[:obj:`InstanceData`]: Detection results of each image
                after the post process. Each item usually contains following
                keys:
                    - scores (Tensor): Classification scores, has a shape
                      (num_instance, )
                    - labels (Tensor): Labels of bboxes, has a shape
                      (num_instances, ).
            """
            proposals = [res.bboxes for res in rpn_results_list]
            rois = bbox2roi(proposals)
            bbox_results = self._bbox_forward(x, rois, batch_img_metas)

            # split batch bbox prediction back to each image
            cls_scores = bbox_results['cls_score']
            num_proposals_per_img = tuple(len(p) for p in proposals)
            rois = rois.split(num_proposals_per_img, 0)
            cls_scores = cls_scores.split(num_proposals_per_img, 0)

            result_list = self.bbox_head.predict_by_feat(
                rois=rois,
                cls_scores=cls_scores,
                batch_img_metas=batch_img_metas,
                rcnn_test_cfg=rcnn_test_cfg)

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
