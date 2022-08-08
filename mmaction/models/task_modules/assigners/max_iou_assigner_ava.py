# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import Tensor

try:
    from mmdet.models.task_modules import AssignResult, MaxIoUAssigner
    from mmdet.registry import TASK_UTILS as MMDET_TASK_UTILS
    mmdet_imported = True
except (ImportError, ModuleNotFoundError):
    mmdet_imported = False

if mmdet_imported:

    @MMDET_TASK_UTILS.register_module()
    class MaxIoUAssignerAVA(MaxIoUAssigner):
        """Assign a corresponding gt bbox or background to each bbox.

        Each proposals will be assigned with `-1`, `0`, or a positive integer
        indicating the ground truth index.

        - -1: don't care
        - 0: negative sample, no assigned gt
        - positive integer: positive sample, index (1-based) of assigned gt

        Args:
            pos_iou_thr (float): IoU threshold for positive bboxes.
            neg_iou_thr (float | tuple): IoU threshold for negative bboxes.
            min_pos_iou (float): Minimum iou for a bbox to be considered as a
                positive bbox. Positive samples can have smaller IoU than
                pos_iou_thr due to the 4th step (assign max IoU sample to each
                gt). Defaults to 0.
            gt_max_assign_all (bool): Whether to assign all bboxes with the
                same highest overlap with some gt to that gt. Defaults to True.
        """

        # The function is overridden, to handle the case that gt_label is not
        # int
        def assign_wrt_overlaps(self, overlaps: Tensor,
                                gt_labels: Tensor) -> AssignResult:
            """Assign w.r.t. the overlaps of bboxes with gts.

            Args:
                overlaps (Tensor): Overlaps between k gt_bboxes and n bboxes,
                    shape(k, n).
                gt_labels (Tensor): Labels of k gt_bboxes, shape
                    (k, num_classes).

            Returns:
                :obj:`AssignResult`: The assign result.
            """
            num_gts, num_bboxes = overlaps.size(0), overlaps.size(1)

            # 1. assign -1 by default
            assigned_gt_inds = overlaps.new_full((num_bboxes, ),
                                                 -1,
                                                 dtype=torch.long)

            if num_gts == 0 or num_bboxes == 0:
                # No ground truth or boxes, return empty assignment
                max_overlaps = overlaps.new_zeros((num_bboxes, ))
                assigned_labels = overlaps.new_full((num_bboxes, ),
                                                    -1,
                                                    dtype=torch.long)
                if num_gts == 0:
                    # No truth, assign everything to background
                    assigned_gt_inds[:] = 0
                return AssignResult(
                    num_gts=num_gts,
                    gt_inds=assigned_gt_inds,
                    max_overlaps=max_overlaps,
                    labels=assigned_labels)

            # for each anchor, which gt best overlaps with it
            # for each anchor, the max iou of all gts
            max_overlaps, argmax_overlaps = overlaps.max(dim=0)
            # for each gt, which anchor best overlaps with it
            # for each gt, the max iou of all proposals
            gt_max_overlaps, gt_argmax_overlaps = overlaps.max(dim=1)

            # 2. assign negative: below
            # the negative inds are set to be 0
            if isinstance(self.neg_iou_thr, float):
                assigned_gt_inds[(max_overlaps >= 0)
                                 & (max_overlaps < self.neg_iou_thr)] = 0
            elif isinstance(self.neg_iou_thr, tuple):
                assert len(self.neg_iou_thr) == 2
                assigned_gt_inds[(max_overlaps >= self.neg_iou_thr[0])
                                 & (max_overlaps < self.neg_iou_thr[1])] = 0

            # 3. assign positive: above positive IoU threshold
            pos_inds = max_overlaps >= self.pos_iou_thr
            assigned_gt_inds[pos_inds] = argmax_overlaps[pos_inds] + 1

            if self.match_low_quality:
                # Low-quality matching will overwrite the assigned_gt_inds
                # assigned in Step 3. Thus, the assigned gt might not be the
                # best one for prediction.
                # For example, if bbox A has 0.9 and 0.8 iou with GT bbox
                # 1 & 2, bbox 1 will be assigned as the best target for bbox A
                # in step 3. However, if GT bbox 2's gt_argmax_overlaps = A,
                # bbox A's assigned_gt_inds will be overwritten to be bbox B.
                # This might be the reason that it is not used in ROI Heads.
                for i in range(num_gts):
                    if gt_max_overlaps[i] >= self.min_pos_iou:
                        if self.gt_max_assign_all:
                            max_iou_inds = overlaps[i, :] == gt_max_overlaps[i]
                            assigned_gt_inds[max_iou_inds] = i + 1
                        else:
                            assigned_gt_inds[gt_argmax_overlaps[i]] = i + 1

            # consider multi-class case (AVA)
            assert len(gt_labels[0]) > 1
            assigned_labels = assigned_gt_inds.new_zeros(
                (num_bboxes, len(gt_labels[0])), dtype=torch.float32)

            # If not assigned, labels will be all 0
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]

            return AssignResult(
                num_gts=num_gts,
                gt_inds=assigned_gt_inds,
                max_overlaps=max_overlaps,
                labels=assigned_labels)

else:
    # define an empty class, so that can be imported
    class MaxIoUAssignerAVA:

        def __init__(self, *args, **kwargs):
            raise ImportError(
                'Failed to import `AssignResult`, `MaxIoUAssigner` from '
                '`mmdet.core.bbox` or failed to import `TASK_UTILS` from '
                '`mmdet.registry`. The class `MaxIoUAssignerAVA` is '
                'invalid. ')
