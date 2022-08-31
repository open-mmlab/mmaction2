# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
from collections import OrderedDict
from typing import Any, Optional, Sequence, Tuple

import mmcv
import mmengine
import numpy as np
from mmengine.evaluator import BaseMetric

from mmaction.evaluation import average_recall_at_avg_proposals
from mmaction.registry import METRICS
from mmaction.utils import ConfigType


@METRICS.register_module()
class ANetMetric(BaseMetric):
    """ActivityNet dataset evaluation metric."""

    def __init__(self,
                 metric_type: str = 'TEM',
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 metric_options: dict = {},
                 dump_config: ConfigType = dict(out='')):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.metric_type = metric_type

        assert 'out' in dump_config
        self.output_format = dump_config.pop('output_format', 'csv')
        self.out = dump_config['out']

        self.metric_options = metric_options
        if self.metric_type == 'AR@AN':
            self.ground_truth = {}

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

        if self.metric_type == 'AR@AN':
            data_batch = data_batch['data_samples']
            for data_sample in data_batch:
                video_info = data_sample.metainfo
                video_id = video_info['video_name'][2:]
                this_video_gt = []
                for ann in video_info['annotations']:
                    t_start, t_end = ann['segment']
                    label = ann['label']
                    this_video_gt.append([t_start, t_end, label])
                self.ground_truth[video_id] = np.array(this_video_gt)

    def compute_metrics(self, results: list) -> dict:
        """Compute the metrics from processed results.

        If `metric_type` is 'TEM', only dump middle results and do not compute
        any metrics.
        Args:
            results (list): The processed results of each batch.
        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        self.dump_results(results)
        if self.metric_type == 'AR@AN':
            return self.compute_ARAN(results)
        return OrderedDict()

    def compute_ARAN(self, results: list) -> dict:
        """AR@AN evaluation metric."""
        temporal_iou_thresholds = self.metric_options.setdefault(
            'AR@AN', {}).setdefault('temporal_iou_thresholds',
                                    np.linspace(0.5, 0.95, 10))
        max_avg_proposals = self.metric_options.setdefault(
            'AR@AN', {}).setdefault('max_avg_proposals', 100)
        if isinstance(temporal_iou_thresholds, list):
            temporal_iou_thresholds = np.array(temporal_iou_thresholds)

        eval_results = OrderedDict()
        proposal, num_proposals = self._import_proposals(results)

        recall, _, _, auc = average_recall_at_avg_proposals(
            self.ground_truth,
            proposal,
            num_proposals,
            max_avg_proposals=max_avg_proposals,
            temporal_iou_thresholds=temporal_iou_thresholds)
        eval_results['auc'] = auc
        eval_results['AR@1'] = np.mean(recall[:, 0])
        eval_results['AR@5'] = np.mean(recall[:, 4])
        eval_results['AR@10'] = np.mean(recall[:, 9])
        eval_results['AR@100'] = np.mean(recall[:, 99])

        return eval_results

    def dump_results(self, results, version='VERSION 1.3'):
        """Save middle or final results to disk."""
        if self.output_format == 'json':
            result_dict = self.proposals2json(results)
            output_dict = {
                'version': version,
                'results': result_dict,
                'external_data': {}
            }
            mmengine.dump(output_dict, self.out)
        elif self.output_format == 'csv':
            os.makedirs(self.out, exist_ok=True)
            header = 'action,start,end,tmin,tmax'
            for result in results:
                video_name, outputs = result
                output_path = osp.join(self.out, video_name + '.csv')
                np.savetxt(
                    output_path,
                    outputs,
                    header=header,
                    delimiter=',',
                    comments='')
        else:
            raise ValueError(
                f'The output format {self.output_format} is not supported.')

    @staticmethod
    def proposals2json(results, show_progress=False):
        """Convert all proposals to a final dict(json) format.
        Args:
            results (list[dict]): All proposals.
            show_progress (bool): Whether to show the progress bar.
                Defaults: False.
        Returns:
            dict: The final result dict. E.g.
            .. code-block:: Python
                dict(video-1=[dict(segment=[1.1,2.0]. score=0.9),
                              dict(segment=[50.1, 129.3], score=0.6)])
        """
        result_dict = {}
        print('Convert proposals to json format')
        if show_progress:
            prog_bar = mmcv.ProgressBar(len(results))
        for result in results:
            video_name = result['video_name']
            result_dict[video_name[2:]] = result['proposal_list']
            if show_progress:
                prog_bar.update()
        return result_dict

    @staticmethod
    def _import_proposals(results):
        """Read predictions from results."""
        proposals = {}
        num_proposals = 0
        for result in results:
            video_id = result['video_name'][2:]
            this_video_proposals = []
            for proposal in result['proposal_list']:
                t_start, t_end = proposal['segment']
                score = proposal['score']
                this_video_proposals.append([t_start, t_end, score])
                num_proposals += 1
            proposals[video_id] = np.array(this_video_proposals)
        return proposals, num_proposals
