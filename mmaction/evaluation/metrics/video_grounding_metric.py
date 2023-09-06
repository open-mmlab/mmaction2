# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Optional, Sequence, Tuple

from mmengine.evaluator import BaseMetric

from mmaction.registry import METRICS


@METRICS.register_module()
class RecallatTopK(BaseMetric):
    """ActivityNet dataset evaluation metric."""

    def __init__(self,
                 topK_list: Tuple[int] = (1, 5),
                 threshold: float = 0.5,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.topK_list = topK_list
        self.threshold = threshold

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

    def compute_metrics(self, results: list) -> dict:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.
        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        eval_results = dict()
        for topK in self.topK_list:
            total = len(results)
            correct = 0.0
            for result in results:
                gt = result['gt']
                predictions = result['predictions'][:topK]
                for prediction in predictions:
                    IoU = self.calculate_IoU(gt, prediction)
                    if IoU > self.threshold:
                        correct += 1
                        break
            acc = correct / total
            eval_results[f'Recall@Top{topK}_IoU={self.threshold}'] = acc
        return eval_results

    def calculate_IoU(self, i0, i1):
        union = (min(i0[0], i1[0]), max(i0[1], i1[1]))
        inter = (max(i0[0], i1[0]), min(i0[1], i1[1]))
        iou = (inter[1] - inter[0]) / (union[1] - union[0])
        return iou
