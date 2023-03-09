# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Optional, Sequence, Tuple

import numpy as np
from mmengine import load
from mmengine.evaluator import BaseMetric

from mmaction.evaluation import frameAP, link_tubes, videoAP, videoAP_all
from mmaction.registry import METRICS


@METRICS.register_module()
class MultiSportsMetric(BaseMetric):
    """MAP Metric for MultiSports dataset."""
    default_prefix: Optional[str] = 'mAP'

    def __init__(self,
                 ann_file: str,
                 metric_options: Optional[dict] = dict(
                     F_mAP=dict(thr=(0.5)),
                     V_mAP=dict(thr=(0.2, 0.5), all=True, tube_thr=15)),
                 collect_device: str = 'cpu',
                 verbose: bool = True,
                 prefix: Optional[str] = None):
        super().__init__(collect_device=collect_device, prefix=prefix)

        self.metric_options = metric_options
        self.annos = load(ann_file)
        self.verbose = verbose

    def process(self, data_batch: Sequence[Tuple[Any, dict]],
                data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (Sequence[Tuple[Any, dict]]): A batch of data
                from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from
                the model.
        """

        for pred in data_samples:
            video_key = pred['video_id'].split('.mp4')[0]
            frm_num = pred['timestamp']
            bboxes = pred['pred_instances']['bboxes'].cpu().numpy()
            cls_scores = pred['pred_instances']['scores'].cpu().numpy()
            det_result = [video_key, frm_num, bboxes, cls_scores]

            self.results.append(det_result)

    def compute_metrics(self, results: list) -> dict:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.
        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        test_videos = self.annos['test_videos'][0]
        resolutions = self.annos['resolution']
        detections = []
        for result in results:
            video_key, frm_num, bboxes, cls_scores = result
            for bbox, cls_score in zip(bboxes, cls_scores):
                video_idx = test_videos.index(video_key)
                pred_label = np.argmax(cls_score)
                score = cls_score[pred_label]
                h, w = resolutions[video_key]
                bbox *= np.array([w, h, w, h])
                instance_result = np.array(
                    [video_idx, frm_num, pred_label, score, *bbox])
                detections.append(instance_result)

        frm_detections = np.array(detections)

        metric_result = dict()
        f_map = frameAP(self.annos, frm_detections,
                        self.metric_options['F_mAP']['thr'], self.verbose)
        metric_result.update({'frameAP': round(f_map, 4)})
        video_tubes = link_tubes(
            self.annos,
            frm_detections,
            len_thre=self.metric_options['V_mAP']['tube_thr'])

        v_map = {}
        for thr in self.metric_options['V_mAP']['thr']:
            map = videoAP(
                self.annos, video_tubes, thr, print_info=self.verbose)
            v_map.update({f'v_map@{thr}': round(map, 4)})
            metric_result.update(v_map)
        if self.metric_options['V_mAP'].get('all'):
            all_map = videoAP_all(self.annos, video_tubes)
            metric_result.update(all_map)
        return metric_result
