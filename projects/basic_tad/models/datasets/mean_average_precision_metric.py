# Copyright (c) OpenMMLab. All rights reserved.
import copy
from collections import OrderedDict
from typing import Optional, Sequence

import numpy as np
from mmaction.registry import METRICS
# from mmdet.evaluation.functional import eval_map
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
from mmengine.structures import InstanceData

from ..models.task_modules.segments_ops import batched_nms1d
from ..models.task_modules.segments_ops import eval_map


@METRICS.register_module()
class mAP(BaseMetric):
    default_prefix: Optional[str] = 'att'

    def __init__(self,
                 iou_thrs=[0.3, 0.4, 0.5, 0.6, 0.7],
                 scale_ranges=None,
                 nms_cfg=dict(type='nms', iou_thr=0.5),
                 max_per_video=1200,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.iou_thrs = [iou_thrs] if isinstance(iou_thrs, float) \
            else iou_thrs
        self.scale_ranges = scale_ranges
        self.nms_cfg = nms_cfg
        self.max_per_video = max_per_video

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.
        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of data samples that
                contain annotations and predictions.
        """
        for data_sample in data_samples:
            gt = copy.deepcopy(data_sample)
            # TODO: Need to refactor to support LoadAnnotations
            gt_instances = gt['gt_instances']
            gt_ignore_instances = gt['ignored_instances']
            ann = dict(
                video_name=gt.get('video_name'),
                labels=gt_instances['labels'].cpu().numpy(),
                segments=gt_instances['bboxes'].cpu().numpy(),
                segments_ignore=gt_ignore_instances['bboxes'].cpu().numpy(),
                labels_ignore=gt_ignore_instances['labels'].cpu().numpy())

            # change back to InstanceData
            pred = InstanceData(**data_sample['pred_instances'])
            pred['bboxes'] = pred['bboxes'].cpu()
            pred['scores'] = pred['scores'].cpu()
            pred['labels'] = pred['labels'].cpu()

            self.results.append((ann, pred))

    def compute_metrics(self, results: list) -> dict:
        """Compute the metrics from processed results.
        Args:
            results (list): The processed results of each batch.
        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        gts, preds = zip(*results)
        gts, preds = self.merge_results_of_same_video(gts, preds)
        logger.info(f'\nConducting the Non-maximum suppression ...')
        preds = self.non_maximum_suppression(preds)
        eval_results = OrderedDict()
        assert isinstance(self.iou_thrs, list)

        mean_aps = []
        for iou_thr in self.iou_thrs:
            logger.info(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
            # Follow the official implementation,
            # http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar
            # we should use the legacy coordinate system in mmdet 1.x,
            # which means w, h should be computed as 'x2 - x1 + 1` and
            # `y2 - y1 + 1`
            mean_ap, _ = eval_map(
                preds,
                gts,
                scale_ranges=self.scale_ranges,
                iou_thr=iou_thr,
                logger=logger,
                mode='anet',
                label_names=self.dataset_meta['classes'])
            mean_aps.append(mean_ap)
            eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)
        eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
        eval_results.move_to_end('mAP', last=False)
        return eval_results

    @staticmethod
    def merge_results_of_same_video(gts, preds):
        video_names = [gt['video_name'] for gt in gts]
        video_names = list(set([vn.rsplit('.', 1)[0] for vn in video_names]))

        merged_gts_dict = dict()
        merged_preds_dict = dict()
        for this_gt, this_pred in zip(gts, preds):
            for vn in video_names:
                if this_gt['video_name'].rsplit('.', 1)[0] == vn:
                    merged_preds_dict.setdefault(vn, []).append(this_pred)
                    merged_gts_dict.setdefault(vn, this_gt)
                    break
            else:
                raise TypeError(
                    f"The gt of {this_gt['video_name']} cannot be categorised into one of the {video_names}")

        # dict of list to list of dict
        merged_gts = []
        merged_preds = []
        for vn in video_names:
            merged_gts.append(merged_gts_dict[vn])
            # for i in merged_preds_dict[vn]:
            #     print(type(i))
            merged_preds.append(InstanceData.cat(merged_preds_dict[vn]))
            # concat_preds = np.concatenate(merged_preds[vn], axis=0)
            # bboxes, scores, labels = np.split(concat_preds, [2, 3], axis=1)
            # merged_preds.append([bboxes, np.squeeze(scores, axis=1), np.squeeze(labels, axis=1)])
        return merged_gts, merged_preds

    def non_maximum_suppression(self, preds):
        preds_nms = []
        for pred_v in preds:
            bboxes, keep_idxs = batched_nms1d(pred_v.bboxes,
                                              pred_v.scores,
                                              pred_v.labels,
                                              nms_cfg=self.nms_cfg)
            pred_v = pred_v[keep_idxs]
            # some nms operation may reweight the score such as softnms
            pred_v.scores = bboxes[:, -1]
            pred_v = pred_v[:self.max_per_video]

            dets = []
            for label in range(len(self.dataset_meta['classes'])):
                index = np.where(pred_v.labels == label)[0]
                pred_bbox_scores = np.hstack(
                    [pred_v[index].bboxes, pred_v[index].scores.reshape((-1, 1))])
                dets.append(pred_bbox_scores)

            preds_nms.append(dets)
        return preds_nms
