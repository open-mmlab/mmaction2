# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Optional, Sequence, Tuple, Union

from mmeval import Accuracy as MMEVAL_Accuracy

from mmaction.registry import METRICS


@METRICS.register_module()
class AccMetric(MMEVAL_Accuracy):
    default_prefix: Optional[str] = 'acc'

    def __init__(self, topk: Union[int, Sequence[int]] = (1, 5)):
        super(AccMetric, self).__init__(topk=topk)

    def process(self, data_batch: Sequence[Tuple[Any, dict]],
                data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and data_samples. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (Sequence[Tuple[Any, dict]]): A batch of data
                from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from
                the model.
        """
        predictions = []
        labels = []
        for data_sample in data_samples:
            pred = data_sample['pred_scores']
            label = data_sample['gt_labels']
            predictions.append(pred['item'])
            labels.append(label['item'])
        self.add(predictions, labels)

    def evaluate(self, *args, **kwargs) -> dict:
        """Returns metric results and reset state.

        This method would be invoked by ``mmengine.Evaluator``.
        """
        metric_results = self.compute(*args, **kwargs)
        self.reset()

        return metric_results
