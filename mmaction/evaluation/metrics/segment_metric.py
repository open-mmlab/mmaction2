# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from typing import Any, Optional, Sequence, Tuple

import numpy as np
from mmengine.evaluator import BaseMetric

from mmaction.registry import METRICS
from mmaction.utils import ConfigType


@METRICS.register_module()
class SegmentMetric(BaseMetric):
    """Action Segmentation dataset evaluation metric."""

    def __init__(self,
                 metric_type: str = 'ALL',
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 metric_options: dict = {},
                 dump_config: ConfigType = dict(out='')):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.metric_type = metric_type

        assert metric_type == 'ALL'
        assert 'out' in dump_config
        self.output_format = dump_config.pop('output_format', 'csv')
        self.out = dump_config['out']

        self.metric_options = metric_options

    def process(self, data_batch: Sequence[Tuple[Any, dict]],
                predictions: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (Sequence[Tuple[Any, dict]]): A batch of data
                from the dataloader.
            predictions (Sequence[dict]): A batch of outputs from
                the model.
        """
        for pred in predictions:
            self.results.append(pred)

        if self.metric_type == 'ALL':
            data_batch = data_batch['data_samples']

    def compute_metrics(self, results: list) -> dict:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.
        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        if self.metric_type == 'ALL':
            return self.compute_ALL(results)
        return OrderedDict()

    def compute_ALL(self, results: list) -> dict:
        """ALL evaluation metric."""
        eval_results = OrderedDict()
        overlap = [.1, .25, .5]
        tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)

        correct = 0
        total = 0
        edit = 0

        for vid in self.results:

            gt_content = vid['ground']
            recog_content = vid['recognition']

            for i in range(len(gt_content)):
                total += 1
                if gt_content[i] == recog_content[i]:
                    correct += 1

            edit += self.edit_score(recog_content, gt_content)

            for s in range(len(overlap)):
                tp1, fp1, fn1 = self.f_score(recog_content, gt_content,
                                             overlap[s])
                tp[s] += tp1
                fp[s] += fp1
                fn[s] += fn1
        eval_results['Acc'] = 100 * float(correct) / total
        eval_results['Edit'] = (1.0 * edit) / len(self.results)
        f1s = np.array([0, 0, 0], dtype=float)
        for s in range(len(overlap)):
            precision = tp[s] / float(tp[s] + fp[s])
            recall = tp[s] / float(tp[s] + fn[s])

            f1 = 2.0 * (precision * recall) / (precision + recall)

            f1 = np.nan_to_num(f1) * 100
            f1s[s] = f1

        eval_results['F1@10'] = f1s[0]
        eval_results['F1@25'] = f1s[1]
        eval_results['F1@50'] = f1s[2]

        return eval_results

    def f_score(self,
                recognized,
                ground_truth,
                overlap,
                bg_class=['background']):
        p_label, p_start, p_end = self.get_labels_start_end_time(
            recognized, bg_class)
        y_label, y_start, y_end = self.get_labels_start_end_time(
            ground_truth, bg_class)

        tp = 0
        fp = 0

        hits = np.zeros(len(y_label))

        for j in range(len(p_label)):
            intersection = np.minimum(p_end[j], y_end) - np.maximum(
                p_start[j], y_start)
            union = np.maximum(p_end[j], y_end) - np.minimum(
                p_start[j], y_start)
            IoU = (1.0 * intersection / union) * (
                [p_label[j] == y_label[x] for x in range(len(y_label))])
            # Get the best scoring segment
            idx = np.array(IoU).argmax()

            if IoU[idx] >= overlap and not hits[idx]:
                tp += 1
                hits[idx] = 1
            else:
                fp += 1
        fn = len(y_label) - sum(hits)
        return float(tp), float(fp), float(fn)

    def edit_score(self,
                   recognized,
                   ground_truth,
                   norm=True,
                   bg_class=['background']):
        P, _, _ = self.get_labels_start_end_time(recognized, bg_class)
        Y, _, _ = self.get_labels_start_end_time(ground_truth, bg_class)
        return self.levenstein(P, Y, norm)

    def get_labels_start_end_time(self,
                                  frame_wise_labels,
                                  bg_class=['background']):
        labels = []
        starts = []
        ends = []
        last_label = frame_wise_labels[0]
        if frame_wise_labels[0] not in bg_class:
            labels.append(frame_wise_labels[0])
            starts.append(0)
        for i in range(len(frame_wise_labels)):
            if frame_wise_labels[i] != last_label:
                if frame_wise_labels[i] not in bg_class:
                    labels.append(frame_wise_labels[i])
                    starts.append(i)
                if last_label not in bg_class:
                    ends.append(i)
                last_label = frame_wise_labels[i]
        if last_label not in bg_class:
            ends.append(i)
        return labels, starts, ends

    def levenstein(self, p, y, norm=False):
        m_row = len(p)
        n_col = len(y)
        D = np.zeros([m_row + 1, n_col + 1], np.float64)
        for i in range(m_row + 1):
            D[i, 0] = i
        for i in range(n_col + 1):
            D[0, i] = i

        for j in range(1, n_col + 1):
            for i in range(1, m_row + 1):
                if y[j - 1] == p[i - 1]:
                    D[i, j] = D[i - 1, j - 1]
                else:
                    D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1,
                                  D[i - 1, j - 1] + 1)

        if norm:
            score = (1 - D[-1, -1] / max(m_row, n_col)) * 100
        else:
            score = D[-1, -1]

        return score
