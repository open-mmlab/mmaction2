# Copyright (c) OpenMMLab. All rights reserved.
import os
from datetime import datetime
from typing import Any, List, Optional, Sequence, Tuple

from mmengine.evaluator import BaseMetric

from mmaction.evaluation import ava_eval, results2csv
from mmaction.registry import METRICS
from mmaction.structures import bbox2result


@METRICS.register_module()
class AVAMetric(BaseMetric):
    """AVA evaluation metric."""
    default_prefix: Optional[str] = 'mAP'

    def __init__(self,
                 ann_file: str,
                 exclude_file: str,
                 label_file: str,
                 options: Tuple[str] = ('mAP', ),
                 action_thr: float = 0.002,
                 num_classes: int = 81,
                 custom_classes: Optional[List[int]] = None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None):
        super().__init__(collect_device=collect_device, prefix=prefix)
        assert len(options) == 1
        self.ann_file = ann_file
        self.exclude_file = exclude_file
        self.label_file = label_file
        self.num_classes = num_classes
        self.options = options
        self.action_thr = action_thr
        self.custom_classes = custom_classes
        if custom_classes is not None:
            self.custom_classes = list([0] + custom_classes)

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
            self.results.append(result)

    def compute_metrics(self, results: list) -> dict:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.
        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        time_now = datetime.now().strftime('%Y%m%d_%H%M%S')
        temp_file = f'AVA_{time_now}_result.csv'
        results2csv(results, temp_file, self.custom_classes)

        eval_results = ava_eval(
            temp_file,
            self.options[0],
            self.label_file,
            self.ann_file,
            self.exclude_file,
            ignore_empty_frames=True,
            custom_classes=self.custom_classes)

        os.remove(temp_file)

        return eval_results
