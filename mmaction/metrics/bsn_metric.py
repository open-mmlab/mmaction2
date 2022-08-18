# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
from collections import OrderedDict
from typing import Any, Optional, Sequence, Tuple

import mmcv
import numpy as np
from mmengine.evaluator import BaseMetric

from mmaction.registry import METRICS
from mmaction.utils import ConfigType


@METRICS.register_module()
class BSNMetric(BaseMetric):
    """BSN evaluation metric."""

    def __init__(self,
                 metric_type: str = 'TEM',
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 dump_config: ConfigType = dict(out='')):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.metric_type = metric_type

        assert 'out' in dump_config
        self.output_format = dump_config.pop('output_format', 'csv')
        self.out = dump_config['out']
        os.makedirs(self.out, exist_ok=True)

    def process(self, data_batch: Sequence[Tuple[Any, dict]],
                predictions: Sequence[dict]) -> None:
        if self.metric_type == 'TEM':
            for pred in predictions:
                self.results.append(pred)

    def compute_metrics(self, results: list) -> dict:
        if self.metric_type == 'TEM':
            self.dump_results(results)
            return OrderedDict()

    def dump_results(self, results, version='VERSION 1.3'):
        if self.output_format == 'json':
            result_dict = self.proposals2json(results)
            output_dict = {
                'version': version,
                'results': result_dict,
                'external_data': {}
            }
            mmcv.dump(output_dict, self.out)
        elif self.output_format == 'csv':
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
