# Copyright (c) OpenMMLab. All rights reserved.
import copy
from collections import OrderedDict
from itertools import product
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import mmengine
import numpy as np
import torch
from mmengine.evaluator import BaseMetric

from mmaction.evaluation import (get_weighted_score, mean_average_precision,
                                 mean_class_accuracy,
                                 mmit_mean_average_precision, top_k_accuracy)
from mmaction.registry import METRICS


def to_tensor(value):
    """Convert value to torch.Tensor."""
    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value)
    elif isinstance(value, Sequence) and not mmengine.is_str(value):
        value = torch.tensor(value)
    elif not isinstance(value, torch.Tensor):
        raise TypeError(f'{type(value)} is not an available argument.')
    return value


@METRICS.register_module()
class AccMetric(BaseMetric):
    """Accuracy evaluation metric."""
    default_prefix: Optional[str] = 'acc'

    def __init__(self,
                 metric_list: Optional[Union[str, Tuple[str]]] = (
                     'top_k_accuracy', 'mean_class_accuracy'),
                 collect_device: str = 'cpu',
                 metric_options: Optional[Dict] = dict(
                     top_k_accuracy=dict(topk=(1, 5))),
                 prefix: Optional[str] = None) -> None:

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

        self.metrics = metrics
        self.metric_options = metric_options

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
            pred = data_sample['pred_score']
            label = data_sample['gt_label']

            # Ad-hoc for RGBPoseConv3D
            if isinstance(pred, dict):
                for item_name, score in pred.items():
                    pred[item_name] = score.cpu().numpy()
            else:
                pred = pred.cpu().numpy()

            result['pred'] = pred
            if label.size(0) == 1:
                # single-label
                result['label'] = label.item()
            else:
                # multi-label
                result['label'] = label.cpu().numpy()
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

        eval_results = dict()
        # Ad-hoc for RGBPoseConv3D
        if isinstance(results[0]['pred'], dict):

            for item_name in results[0]['pred'].keys():
                preds = [x['pred'][item_name] for x in results]
                eval_result = self.calculate(preds, labels)
                eval_results.update(
                    {f'{item_name}_{k}': v
                     for k, v in eval_result.items()})

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

        # Simple Acc Calculation
        else:
            preds = [x['pred'] for x in results]
            return self.calculate(preds, labels)

    def calculate(self, preds: List[np.ndarray],
                  labels: List[Union[int, np.ndarray]]) -> Dict:
        """Compute the metrics from processed results.

        Args:
            preds (list[np.ndarray]): List of the prediction scores.
            labels (list[int | np.ndarray]): List of the labels.

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
                if metric == 'mean_average_precision':
                    mAP = mean_average_precision(preds, labels)
                    eval_results['mean_average_precision'] = mAP

                elif metric == 'mmit_mean_average_precision':
                    mAP = mmit_mean_average_precision(preds, labels)
                    eval_results['mmit_mean_average_precision'] = mAP

        return eval_results


@METRICS.register_module()
class ConfusionMatrix(BaseMetric):
    r"""A metric to calculate confusion matrix for single-label tasks.

    Args:
        num_classes (int, optional): The number of classes. Defaults to None.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.

    Examples:

        1. The basic usage.

        >>> import torch
        >>> from mmaction.evaluation import ConfusionMatrix
        >>> y_pred = [0, 1, 1, 3]
        >>> y_true = [0, 2, 1, 3]
        >>> ConfusionMatrix.calculate(y_pred, y_true, num_classes=4)
        tensor([[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1]])
        >>> # plot the confusion matrix
        >>> import matplotlib.pyplot as plt
        >>> y_score = torch.rand((1000, 10))
        >>> y_true = torch.randint(10, (1000, ))
        >>> matrix = ConfusionMatrix.calculate(y_score, y_true)
        >>> ConfusionMatrix().plot(matrix)
        >>> plt.show()

        2. In the config file

        .. code:: python

            val_evaluator = dict(type='ConfusionMatrix')
            test_evaluator = dict(type='ConfusionMatrix')
    """  # noqa: E501
    default_prefix = 'confusion_matrix'

    def __init__(self,
                 num_classes: Optional[int] = None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device, prefix)

        self.num_classes = num_classes

    def process(self, data_batch, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            pred_scores = data_sample.get('pred_score')
            gt_label = data_sample['gt_label']
            if pred_scores is not None:
                pred_label = pred_scores.argmax(dim=0, keepdim=True)
                self.num_classes = pred_scores.size(0)
            else:
                pred_label = data_sample['pred_label']

            self.results.append({
                'pred_label': pred_label,
                'gt_label': gt_label
            })

    def compute_metrics(self, results: list) -> dict:
        pred_labels = []
        gt_labels = []
        for result in results:
            pred_labels.append(result['pred_label'])
            gt_labels.append(result['gt_label'])
        confusion_matrix = ConfusionMatrix.calculate(
            torch.cat(pred_labels),
            torch.cat(gt_labels),
            num_classes=self.num_classes)
        return {'result': confusion_matrix}

    @staticmethod
    def calculate(pred, target, num_classes=None) -> dict:
        """Calculate the confusion matrix for single-label task.

        Args:
            pred (torch.Tensor | np.ndarray | Sequence): The prediction
                results. It can be labels (N, ), or scores of every
                class (N, C).
            target (torch.Tensor | np.ndarray | Sequence): The target of
                each prediction with shape (N, ).
            num_classes (Optional, int): The number of classes. If the ``pred``
                is label instead of scores, this argument is required.
                Defaults to None.

        Returns:
            torch.Tensor: The confusion matrix.
        """
        pred = to_tensor(pred)
        target_label = to_tensor(target).int()

        assert pred.size(0) == target_label.size(0), \
            f"The size of pred ({pred.size(0)}) doesn't match "\
            f'the target ({target_label.size(0)}).'
        assert target_label.ndim == 1

        if pred.ndim == 1:
            assert num_classes is not None, \
                'Please specify the `num_classes` if the `pred` is labels ' \
                'intead of scores.'
            pred_label = pred
        else:
            num_classes = num_classes or pred.size(1)
            pred_label = torch.argmax(pred, dim=1).flatten()

        with torch.no_grad():
            indices = num_classes * target_label + pred_label
            matrix = torch.bincount(indices, minlength=num_classes**2)
            matrix = matrix.reshape(num_classes, num_classes)

        return matrix

    @staticmethod
    def plot(confusion_matrix: torch.Tensor,
             include_values: bool = False,
             cmap: str = 'viridis',
             classes: Optional[List[str]] = None,
             colorbar: bool = True,
             show: bool = True):
        """Draw a confusion matrix by matplotlib.

        Modified from `Scikit-Learn
        <https://github.com/scikit-learn/scikit-learn/blob/dc580a8ef/sklearn/metrics/_plot/confusion_matrix.py#L81>`_

        Args:
            confusion_matrix (torch.Tensor): The confusion matrix to draw.
            include_values (bool): Whether to draw the values in the figure.
                Defaults to False.
            cmap (str): The color map to use. Defaults to use "viridis".
            classes (list[str], optional): The names of categories.
                Defaults to None, which means to use index number.
            colorbar (bool): Whether to show the colorbar. Defaults to True.
            show (bool): Whether to show the figure immediately.
                Defaults to True.
        """  # noqa: E501
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 10))

        num_classes = confusion_matrix.size(0)

        im_ = ax.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
        text_ = None
        cmap_min, cmap_max = im_.cmap(0), im_.cmap(1.0)

        if include_values:
            text_ = np.empty_like(confusion_matrix, dtype=object)

            # print text with appropriate color depending on background
            thresh = (confusion_matrix.max() + confusion_matrix.min()) / 2.0

            for i, j in product(range(num_classes), range(num_classes)):
                color = cmap_max if confusion_matrix[i,
                                                     j] < thresh else cmap_min

                text_cm = format(confusion_matrix[i, j], '.2g')
                text_d = format(confusion_matrix[i, j], 'd')
                if len(text_d) < len(text_cm):
                    text_cm = text_d

                text_[i, j] = ax.text(
                    j, i, text_cm, ha='center', va='center', color=color)

        display_labels = classes or np.arange(num_classes)

        if colorbar:
            fig.colorbar(im_, ax=ax)
        ax.set(
            xticks=np.arange(num_classes),
            yticks=np.arange(num_classes),
            xticklabels=display_labels,
            yticklabels=display_labels,
            ylabel='True label',
            xlabel='Predicted label',
        )
        ax.invert_yaxis()
        ax.xaxis.tick_top()

        ax.set_ylim((num_classes - 0.5, -0.5))
        # Automatically rotate the x labels.
        fig.autofmt_xdate(ha='center')

        if show:
            plt.show()
        return fig
