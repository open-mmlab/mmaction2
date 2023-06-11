# Copyright (c) OpenMMLab. All rights reserved.
import copy
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from mmengine.evaluator import BaseMetric

from mmaction.registry import METRICS


@METRICS.register_module()
class RetrievalMetric(BaseMetric):
    """Metric for video retrieval task.

    Args:
        metric_list (str | tuple[str]): The list of the metrics to be
            computed. Defaults to ``('R1', 'R5', 'R10', 'MdR', 'MnR')``.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """

    default_prefix = 'retrieval'

    def __init__(self,
                 metric_list: Union[Tuple[str],
                                    str] = ('R1', 'R5', 'R10', 'MdR', 'MnR'),
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        if isinstance(metric_list, str):
            metric_list = (metric_list, )

        for metric in metric_list:
            if metric not in ['R1', 'R5', 'R10', 'MdR', 'MnR']:
                raise ValueError(f'RetrievalMetric only supports '
                                 f"'R1', 'R5', 'R10', 'MdR', 'MnR', "
                                 f"but got '{metric}. '")

        self.metric_list = metric_list

    def process(self, data_batch: Optional[Dict],
                data_samples: Sequence[Dict]) -> None:
        """Process one batch of data samples and data_samples. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (dict, optional): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        data_samples = copy.deepcopy(data_samples)

        for data_sample in data_samples:
            results = dict()
            features = data_sample['features']
            video_feature = features['video_feature'].cpu().numpy()
            text_feature = features['text_feature'].cpu().numpy()
            results['video_feature'] = video_feature
            results['text_feature'] = text_feature
            self.results.append(results)

    def compute_metrics(self, results: List) -> Dict:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """

        video_features = np.stack([res['video_feature'] for res in results])
        text_features = np.stack([res['text_feature'] for res in results])

        video_features = video_features / np.linalg.norm(
            video_features, axis=-1, keepdims=True)
        text_features = text_features / np.linalg.norm(
            text_features, axis=-1, keepdims=True)

        similarity = text_features @ video_features.T

        sx = np.sort(-similarity)
        d = np.diag(-similarity)
        ind = np.where((sx - d[:, None]) == 0)[1]

        metrics = OrderedDict()
        for metric in self.metric_list:
            if metric == 'R1':
                metrics['R1'] = float(np.sum(ind == 0)) * 100 / len(ind)
            elif metric == 'R5':
                metrics['R5'] = float(np.sum(ind < 5)) * 100 / len(ind)
            elif metric == 'R10':
                metrics['R10'] = float(np.sum(ind < 10)) * 100 / len(ind)
            elif metric == 'MdR':
                metrics['MdR'] = np.median(ind) + 1
            elif metric == 'MnR':
                metrics['MnR'] = np.mean(ind) + 1

        return metrics
