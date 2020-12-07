import numpy as np  # isort:skip
from mmdet.models import HEADS  # isort:skip
from mmdet.models.roi_heads import StandardRoIHead  # isort:skip

from mmaction.core.bbox import bbox2result  # isort:skip
from mmdet.core.bbox import bbox2roi  # isort:skip


@HEADS.register_module()
class AVARoIHead(StandardRoIHead):

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False):
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        bbox_results = bbox2result(
            det_bboxes,
            det_labels,
            self.bbox_head.num_classes,
            thr=self.test_cfg.action_thr)
        return [bbox_results]

    def simple_test_bboxes(self, x, img_metas, proposals, rcnn_test_cfg):
        """Test only det bboxes without augmentation."""
        rois = bbox2roi(proposals)
        bbox_results = self._bbox_forward(x, rois)
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
