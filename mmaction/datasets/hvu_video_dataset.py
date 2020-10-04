import os.path as osp

import mmcv
import numpy as np
from mmcv.utils import print_log

from ..core import mean_average_precision
from .base import BaseDataset
from .registry import DATASETS


@DATASETS.register_module()
class HVUVideoDataset(BaseDataset):
    """HVU Video dataset, which support the recognition tags of multiple
    categories.

    The dataset loads raw videos and apply specified transforms to return a
    dict containing the frame tensors and other information.

    The ann_file is a json file with multiple dictionaries, and each dictionary
    indicates a sample video with the filename and tags, the tags are organized
    as different categories. Example of a dictionary:

    .. code-block:: txt

        {
            'filename': 'gD_G1b0wV5I_001015_001035.mp4',
            'label': {
                'concept': [250, 131, 42, 51, 57, 155, 122],
                'object': [1570, 508],
                'event': [16],
                'action': [180],
                'scene': [206]
            }
        }


    Args:
        ann_file (str): Path to the annotation file, should be a json file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        tag_categories (list[str]): List of category names of tags.
        tag_category_nums (list[int]): List of number of tags in each category.
        **kwargs: Keyword arguments for ``BaseDataset``.
    """

    def __init__(self, ann_file, pipeline, tag_categories, tag_category_nums,
                 **kwargs):
        assert len(tag_categories) == len(tag_category_nums)
        self.tag_categories = tag_categories
        self.tag_category_nums = tag_category_nums
        self.num_categories = len(self.tag_categories)
        self.num_tags = sum(self.tag_category_nums)
        self.category2num = {
            k: v
            for k, v in zip(tag_categories, tag_category_nums)
        }
        self.start_idx = [0]
        for i in range(self.num_categories - 1):
            self.start_idx.append(self.start_idx[-1] +
                                  self.tag_category_nums[i])
        self.category2startidx = {
            k: v
            for k, v in zip(tag_categories, self.start_idx)
        }
        super().__init__(ann_file, pipeline, start_index=0, **kwargs)

    def load_annotations(self):
        """Load annotation file to get video information."""
        assert self.ann_file.endswith('.json')
        return self.load_json_annotations()

    def load_json_annotations(self):
        video_infos = mmcv.load(self.ann_file)
        num_videos = len(video_infos)
        path_key = 'filename'

        for i in range(num_videos):
            path_value = video_infos[i][path_key]
            if self.data_prefix is not None:
                path_value = osp.join(self.data_prefix, path_value)
            video_infos[i][path_key] = path_value

            # We will convert label to torch tensors in the pipeline
            video_infos[i]['categories'] = self.tag_categories
            video_infos[i]['category_nums'] = self.tag_category_nums

        return video_infos

    def evaluate(self, results, metrics='mean_average_precision', logger=None):
        """Evaluation in HVU Video Dataset. We only support evaluating mAP for
        each tag categories. Since some tag categories are missing for some
        videos, we can not evaluate mAP for all tags.

        Args:
            results (list): Output results.
            metrics (str | sequence[str]): Metrics to be performed.
                Defaults: 'mean_average_precision'.
            logger (logging.Logger | None): Logger for recording.
                Default: None.

        Return:
            dict: Evaluation results dict.
        """
        if not isinstance(results, list):
            raise TypeError(f'results must be a list, but got {type(results)}')
        assert len(results) == len(self), (
            f'The length of results is not equal to the dataset len: '
            f'{len(results)} != {len(self)}')

        metrics = metrics if isinstance(metrics, (list, tuple)) else [metrics]

        # There should be only one metric in the metrics list:
        # 'mean_average_precision'
        assert len(metrics) == 1
        metric = metrics[0]
        assert metric == 'mean_average_precision'

        gt_labels = [ann['label'] for ann in self.video_infos]

        eval_results = {}
        for i, category in enumerate(self.tag_categories):

            start_idx = self.category2startidx[category]
            num = self.category2num[category]
            preds = [
                result[start_idx:start_idx + num]
                for video_idx, result in enumerate(results)
                if category in gt_labels[video_idx]
            ]
            gts = [
                gt_label[category]
                for video_idx, gt_label in enumerate(gt_labels)
                if category in gt_label
            ]

            # convert label list to ndarray
            def label2array(label):
                arr = np.zeros(num, dtype=np.float32)
                arr[label] = 1.
                return arr

            gts = [label2array(item) for item in gts]

            mAP = mean_average_precision(preds, gts)
            eval_results[f'{category}_mAP'] = mAP
            log_msg = f'\n{category}_mAP\t{mAP:.4f}'
            print_log(log_msg, logger=logger)

        return eval_results
