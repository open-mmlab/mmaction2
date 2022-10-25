# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Sequence, Tuple
from mmaction.registry import METRICS
from mmaction.structures import bbox2result

from mmeval import AVAMeanAP as MMEVAL_AVAMeanAP


@METRICS.register_module()
class AVAMetric(MMEVAL_AVAMeanAP):
    def __init__(self,
                 action_thr: float = 0.002,
                 **kwargs):
        super().__init__(**kwargs)
        self.action_thr = action_thr

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
        for data_sample in data_samples:
            result = dict()
            pred = data_sample['pred_instances']
            result['video_id'] = data_sample['video_id']
            result['timestamp'] = data_sample['timestamp']
            outputs = bbox2result(
                pred['bboxes'],
                pred['scores'],
                num_classes=self.num_classes,
                thr=self.action_thr)
            result['outputs'] = outputs
            self.add(result)

    def evaluate(self, *args, **kwargs) -> dict:
        """Returns metric results and reset state.
        This method would be invoked by ``mmengine.Evaluator``.
        """
        metric_results = self.compute(*args, **kwargs)
        self.reset()

        return metric_results
