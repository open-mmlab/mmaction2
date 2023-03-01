# Copyright (c) OpenMMLab. All rights reserved.
import copy
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from mmengine.evaluator import BaseMetric

from mmaction.evaluation import (get_weighted_score, mean_average_precision,
                                 mean_class_accuracy,
                                 mmit_mean_average_precision, top_k_accuracy)
from mmaction.registry import METRICS


@METRICS.register_module()
class AccMetric(BaseMetric):
    """Accuracy evaluation metric."""
    default_prefix: Optional[str] = 'acc'

    def __init__(
            self,
            metric_list: Optional[Union[str,
                                        Tuple[str]]] = ('top_k_accuracy',
                                                        'mean_class_accuracy'),
            collect_device: str = 'cpu',
            metric_options: Optional[Dict] = dict(
                top_k_accuracy=dict(topk=(1, 5))),
            prefix: Optional[str] = None,
            num_classes: Optional[int] = None):

        # TODO: fix the metric_list argument with a better one.
        # `metrics` is not a safe argument here with mmengine.
        # we have to replace it with `metric_list`.
        super().__init__(collect_device=collect_device, prefix=prefix)
        if not isinstance(metric_list, (str, tuple)):
            raise TypeError('metric_list must be str or tuple of str, '
                            f'but got {type(metric_list)}')

        if isinstance(metric_list, str):
            metrics = (metric_list, )
        else:
            metrics = metric_list

        # coco evaluation metrics
        for metric in metrics:
            assert metric in [
                'top_k_accuracy', 'mean_class_accuracy',
                'mmit_mean_average_precision', 'mean_average_precision'
            ]

            if metric in [
                    'mmit_mean_average_precision', 'mean_average_precision'
            ]:
                assert type(num_classes) == int

        self.metrics = metrics
        self.metric_options = metric_options
        self.num_classes = num_classes

    def process(self, data_batch: Sequence[Tuple[Any, Dict]],
                data_samples: Sequence[Dict]) -> None:
        """Process one batch of data samples and data_samples. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (Sequence[dict]): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        data_samples = copy.deepcopy(data_samples)
        for data_sample in data_samples:
            result = dict()
            pred = data_sample['pred_scores']
            label = data_sample['gt_labels']
            for item_name, score in pred.items():
                pred[item_name] = score.cpu().numpy()
            result['pred'] = pred
            result['label'] = label['item'].item()
            self.results.append(result)

    def compute_metrics(self, results: List) -> Dict:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        labels = [x['label'] for x in results]

        if len(results[0]['pred']) == 1:
            preds = [x['pred']['item'] for x in results]
            return self.calculate(preds, labels)

        eval_results = dict()
        for item_name in results[0]['pred'].keys():
            preds = [x['pred'][item_name] for x in results]
            eval_result = self.calculate(preds, labels)
            eval_results.update(
                {f'{item_name}_{k}': v
                 for k, v in eval_result.items()})

        # Ad-hoc for RGBPoseConv3D
        if len(results[0]['pred']) == 2 and \
                'rgb' in results[0]['pred'] and \
                'pose' in results[0]['pred']:

            rgb = [x['pred']['rgb'] for x in results]
            pose = [x['pred']['pose'] for x in results]

            preds = {
                '1:1': get_weighted_score([rgb, pose], [1, 1]),
                '2:1': get_weighted_score([rgb, pose], [2, 1]),
                '1:2': get_weighted_score([rgb, pose], [1, 2])
            }
            for k in preds:
                eval_result = self.calculate(preds[k], labels)
                eval_results.update({
                    f'RGBPose_{k}_{key}': v
                    for key, v in eval_result.items()
                })

        return eval_results

    def calculate(self, preds: List[np.ndarray], labels: List[int]) -> Dict:
        """Compute the metrics from processed results.

        Args:
            preds (list[np.ndarray]): List of the prediction scores.
            labels (list[int]): List of the labels.

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        eval_results = OrderedDict()
        metric_options = copy.deepcopy(self.metric_options)
        for metric in self.metrics:
            if metric == 'top_k_accuracy':
                topk = metric_options.setdefault('top_k_accuracy',
                                                 {}).setdefault(
                                                     'topk', (1, 5))

                if not isinstance(topk, (int, tuple)):
                    raise TypeError('topk must be int or tuple of int, '
                                    f'but got {type(topk)}')

                if isinstance(topk, int):
                    topk = (topk, )

                top_k_acc = top_k_accuracy(preds, labels, topk)
                for k, acc in zip(topk, top_k_acc):
                    eval_results[f'top{k}'] = acc

            if metric == 'mean_class_accuracy':
                mean1 = mean_class_accuracy(preds, labels)
                eval_results['mean1'] = mean1

            if metric in [
                    'mean_average_precision',
                    'mmit_mean_average_precision',
            ]:
                gt_labels_arrays = [
                    self.label2array(self.num_classes, label)
                    for label in labels
                ]

                if metric == 'mean_average_precision':
                    mAP = mean_average_precision(preds, gt_labels_arrays)
                    eval_results['mean_average_precision'] = mAP

                elif metric == 'mmit_mean_average_precision':
                    mAP = mmit_mean_average_precision(preds, gt_labels_arrays)
                    eval_results['mmit_mean_average_precision'] = mAP

        return eval_results

    @staticmethod
    def label2array(num, label):
        """Convert multi-label to array."""
        arr = np.zeros(num, dtype=np.float32)
        arr[label] = 1.
        return arr
