import copy
from collections import OrderedDict
from typing import Sequence

import numpy as np
import torch
from mmdet.evaluation.functional import eval_map, eval_recalls
from mmdet.evaluation.metrics import VOCMetric
from mmaction.registry import METRICS
from mmdet.structures.bbox import bbox_overlaps
from mmengine.logging import MMLogger
from mmengine.structures import InstanceData
from mmcv.ops import batched_nms


@METRICS.register_module()
class TADmAPMetric(VOCMetric):

    def __init__(self,
                 nms_cfg=dict(type='nms', iou_thr=0.4),
                 max_per_video: int = False,
                 score_thr=0.0,
                 duration_thr=0.0,
                 nms_in_overlap=False,
                 eval_mode: str = 'area',
                 **kwargs):
        super().__init__(eval_mode=eval_mode, **kwargs)
        self.nms_cfg = nms_cfg
        self.max_per_video = max_per_video
        self.score_thr = score_thr
        self.duration_thr = duration_thr
        self.nms_in_overlap = nms_in_overlap

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
            data = copy.deepcopy(data_sample)
            gts, dets = data['gt_instances'], data['pred_instances']
            gts_ignore = data.get('ignored_instances', dict())
            ann = dict(
                video_name=data['img_id'],  # for the purpose of future grouping detections of same video.
                labels=gts['labels'].cpu().numpy(),
                bboxes=gts['bboxes'].cpu().numpy(),
                bboxes_ignore=gts_ignore.get('bboxes', torch.empty((0, 4))).cpu().numpy(),
                labels_ignore=gts_ignore.get('labels', torch.empty(0, )).cpu().numpy())

            if self.nms_in_overlap:
                ann['overlap'] = data['overlap'],  # for the purpose of NMS on overlapped region in testing videos

            # Convert the format of segment predictions from feature-unit to second-unit (add window-offset back first).
            if 'offset_sec' in data:
                dets['bboxes'] = dets['bboxes'] + data['offset_sec']

            # Set y1, y2 of predictions the fixed value.
            dets['bboxes'][:, 1] = 0.1
            dets['bboxes'][:, 3] = 0.9

            # Filter out predictions with low scores
            valid_inds = dets['scores'] > self.score_thr

            # Filter out predictions with short duration
            valid_inds &= (dets['bboxes'][:, 2] - dets['bboxes'][:, 0]) > self.duration_thr

            dets['bboxes'] = dets['bboxes'][valid_inds].cpu()
            dets['scores'] = dets['scores'][valid_inds].cpu()
            dets['labels'] = dets['labels'][valid_inds].cpu()

            # Format predictions to InstanceData
            dets = InstanceData(**dets)

            self.results.append((ann, dets))

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

        # Following the TadTR, we cropped temporally OVERLAPPED sub-videos from the test video
        # to handle test video of long duration while keep a fine temporal granularity.
        # In this case, we need perform non-maximum suppression (NMS) to remove redundant detections.
        # This NMS, however, is NOT necessary when window stride >= window size, i.e., non-overlapped sliding window.
        logger.info(f'\n Concatenating the testing results ...')
        gts, preds = self.merge_results_of_same_video(gts, preds)
        preds = self.non_maximum_suppression(preds)
        eval_results = OrderedDict()
        if self.metric == 'mAP':
            assert isinstance(self.iou_thrs, list)
            dataset_name = self.dataset_meta['classes']

            mean_aps = []
            for iou_thr in self.iou_thrs:
                logger.info(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
                mean_ap, _ = eval_map(
                    preds,
                    gts,
                    scale_ranges=self.scale_ranges,
                    iou_thr=iou_thr,
                    dataset=dataset_name,
                    logger=logger,
                    eval_mode=self.eval_mode,
                    use_legacy_coordinate=False)
                mean_aps.append(mean_ap)
                eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)
            eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
            eval_results.move_to_end('mAP', last=False)
        elif self.metric == 'recall':
            # TODO: Currently not checked.
            gt_bboxes = [ann['bboxes'] for ann in self.annotations]
            recalls = eval_recalls(
                gt_bboxes,
                results,
                self.proposal_nums,
                self.iou_thrs,
                logger=logger,
                use_legacy_coordinate=False)
            for i, num in enumerate(self.proposal_nums):
                for j, iou_thr in enumerate(self.iou_thrs):
                    eval_results[f'recall@{num}@{iou_thr}'] = recalls[i, j]
            if recalls.shape[1] > 1:
                ar = recalls.mean(axis=1)
                for i, num in enumerate(self.proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
        return eval_results

    @staticmethod
    def merge_results_of_same_video(gts, preds):
        # Merge prediction results from the same videos because we use sliding windows to crop the testing videos
        # Also known as the Cross-Window Fusion (CWF)
        video_names = list(dict.fromkeys([gt['video_name'] for gt in gts]))

        merged_gts_dict = dict()
        merged_preds_dict = dict()
        for this_gt, this_pred in zip(gts, preds):
            video_name = this_gt.pop('video_name')
            # Computer the mask indicating that if a prediction is in the overlapped regions.
            overlap_regions = this_gt.pop('overlap', np.empty([0]))
            if overlap_regions.size == 0:
                this_pred.in_overlap = np.zeros(this_pred.bboxes.shape[0], dtype=bool)
            else:
                this_pred.in_overlap = bbox_overlaps(this_pred.bboxes, torch.from_numpy(overlap_regions)) > 0

            merged_preds_dict.setdefault(video_name, []).append(this_pred)
            merged_gts_dict.setdefault(video_name, this_gt)  # the gt is video-wise thus no need concatenation

        # dict of list to list of dict
        merged_gts = []
        merged_preds = []
        for video_name in video_names:
            merged_gts.append(merged_gts_dict[video_name])
            # Concatenate detection in windows of the same video
            merged_preds.append(InstanceData.cat(merged_preds_dict[video_name]))
        return merged_gts, merged_preds

    def non_maximum_suppression(self, preds):
        preds_nms = []
        for pred_v in preds:
            if self.nms_cfg is not None:
                if self.nms_in_overlap:
                    if pred_v.in_overlap.sum() > 1:
                        # Perform NMS over predictions in each overlapped region
                        pred_not_in_overlaps = pred_v[~pred_v.in_overlap.max(-1)[0]]
                        pred_in_overlaps = []
                        for i in range(pred_v.in_overlap.shape[1]):
                            pred_in_overlap = pred_v[pred_v.in_overlap[:, i]]
                            if len(pred_in_overlap) == 0:
                                continue
                            bboxes, keep_idxs = batched_nms(pred_in_overlap.bboxes,
                                                            pred_in_overlap.scores,
                                                            pred_in_overlap.labels,
                                                            nms_cfg=self.nms_cfg)
                            pred_in_overlap = pred_in_overlap[keep_idxs]
                            pred_in_overlap.scores = bboxes[:, -1]
                            pred_in_overlaps.append(pred_in_overlap)
                        pred_v = InstanceData.cat(pred_in_overlaps + [pred_not_in_overlaps])
                else:
                    bboxes, keep_idxs = batched_nms(pred_v.bboxes,
                                                    pred_v.scores,
                                                    pred_v.labels,
                                                    nms_cfg=self.nms_cfg)
                    pred_v = pred_v[keep_idxs]
                    pred_v.scores = bboxes[:, -1]
            sort_idxs = pred_v.scores.argsort(descending=True)
            pred_v = pred_v[sort_idxs]
            # keep top-k predictions
            if self.max_per_video:
                pred_v = pred_v[:self.max_per_video]

            # Reformat predictions to meet the requirement of eval_map function: VideoList[ClassList[PredictionArray]]
            dets = []
            for label in range(len(self.dataset_meta['classes'])):
                index = np.where(pred_v.labels == label)[0]
                pred_bbox_with_scores = np.hstack(
                    [pred_v[index].bboxes, pred_v[index].scores.reshape((-1, 1))])
                dets.append(pred_bbox_with_scores)

            preds_nms.append(dets)
        return preds_nms
