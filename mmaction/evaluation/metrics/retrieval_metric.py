# Copyright (c) OpenMMLab. All rights reserved.
import copy
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from mmengine.evaluator import BaseMetric

from mmaction.registry import METRICS


@METRICS.register_module()
class RetrievalMetric(BaseMetric):

    default_prefix = 'TODO'

    def __init__(self,
                 metric_list: Tuple[str] = ('R1', 'R5', 'R10', 'MdR', 'MnR'),
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        if isinstance(metric_list, str):
            metric_list = (metric_list, )

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
        similarity = text_features @ video_features.T

        sx = np.sort(-similarity)
        d = np.diag(-similarity)
        ind = np.where((sx - d[:, None]) == 0)[1]
        assert len(ind) == similarity.shape[0]

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
            else:
                raise ValueError('')

        return metrics
