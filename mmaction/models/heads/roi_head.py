# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from mmaction.core.bbox import bbox2result
from mmaction.utils import import_module_error_class

try:
    from mmdet.core.bbox import bbox2roi
    from mmdet.models import HEADS as MMDET_HEADS
    from mmdet.models.roi_heads import StandardRoIHead
    mmdet_imported = True
except (ImportError, ModuleNotFoundError):
    mmdet_imported = False

if mmdet_imported:

    @MMDET_HEADS.register_module()
    class AVARoIHead(StandardRoIHead):

        def _bbox_forward(self, x, rois, img_metas):
            """Defines the computation performed to get bbox predictions.

            Args:
                x (torch.Tensor): The input tensor.
                rois (torch.Tensor): The regions of interest.
                img_metas (list): The meta info of images

            Returns:
                dict: bbox predictions with features and classification scores.
            """
            bbox_feat, global_feat = self.bbox_roi_extractor(x, rois)

            if self.with_shared_head:
                bbox_feat = self.shared_head(
                    bbox_feat,
                    feat=global_feat,
                    rois=rois,
                    img_metas=img_metas)

            cls_score, bbox_pred = self.bbox_head(bbox_feat)

            bbox_results = dict(
                cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feat)
            return bbox_results

        def _bbox_forward_train(self, x, sampling_results, gt_bboxes,
                                gt_labels, img_metas):
            """Run forward function and calculate loss for box head in
            training."""
            rois = bbox2roi([res.bboxes for res in sampling_results])
            bbox_results = self._bbox_forward(x, rois, img_metas)

            bbox_targets = self.bbox_head.get_targets(sampling_results,
                                                      gt_bboxes, gt_labels,
                                                      self.train_cfg)
            loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                            bbox_results['bbox_pred'], rois,
                                            *bbox_targets)

            bbox_results.update(loss_bbox=loss_bbox)
            return bbox_results

        def simple_test(self,
                        x,
                        proposal_list,
                        img_metas,
                        proposals=None,
                        rescale=False):
            """Defines the computation performed for simple testing."""
            assert self.with_bbox, 'Bbox head must be implemented.'

            if isinstance(x, tuple):
                x_shape = x[0].shape
            else:
                x_shape = x.shape

            assert x_shape[0] == 1, 'only accept 1 sample at test mode'
            assert x_shape[0] == len(img_metas) == len(proposal_list)

            det_bboxes, det_labels = self.simple_test_bboxes(
                x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
            bbox_results = bbox2result(
                det_bboxes,
                det_labels,
                self.bbox_head.num_classes,
                thr=self.test_cfg.action_thr)
            return [bbox_results]

        def simple_test_bboxes(self,
                               x,
                               img_metas,
                               proposals,
                               rcnn_test_cfg,
                               rescale=False):
            """Test only det bboxes without augmentation."""
            rois = bbox2roi(proposals)
            bbox_results = self._bbox_forward(x, rois, img_metas)
            cls_score = bbox_results['cls_score']

            img_shape = img_metas[0]['img_shape']
            crop_quadruple = np.array([0, 0, 1, 1])
            flip = False

            if 'crop_quadruple' in img_metas[0]:
                crop_quadruple = img_metas[0]['crop_quadruple']

            if 'flip' in img_metas[0]:
                flip = img_metas[0]['flip']

            det_bboxes, det_labels = self.bbox_head.get_det_bboxes(
                rois,
                cls_score,
                img_shape,
                flip=flip,
                crop_quadruple=crop_quadruple,
                cfg=rcnn_test_cfg)

            return det_bboxes, det_labels
else:
    # Just define an empty class, so that __init__ can import it.
    @import_module_error_class('mmdet')
    class AVARoIHead:
        pass
