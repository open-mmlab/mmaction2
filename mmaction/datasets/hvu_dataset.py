import os.path as osp

import mmcv
import numpy as np
from mmcv.utils import print_log

from mmaction.core import build_metrics
from .base import BaseDataset
from .registry import DATASETS


@DATASETS.register_module()
class HVUDataset(BaseDataset):
    """HVU dataset, which supports the recognition tags of multiple categories.
    Accept both video annotation files or rawframe annotation files.

    The dataset loads videos or raw frames and applies specified transforms to
    return a dict containing the frame tensors and other information.

    The ann_file is a json file with multiple dictionaries, and each dictionary
    indicates a sample video with the filename and tags, the tags are organized
    as different categories. Example of a video dictionary:

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

    Example of a rawframe dictionary:

    .. code-block:: txt

        {
            'frame_dir': 'gD_G1b0wV5I_001015_001035',
            'total_frames': 61
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
        filename_tmpl (str | None): Template for each filename. If set to None,
            video dataset is used. Default: None.
        **kwargs: Keyword arguments for ``BaseDataset``.
    """

    def __init__(self,
                 ann_file,
                 pipeline,
                 tag_categories,
                 tag_category_nums,
                 filename_tmpl=None,
                 **kwargs):
        assert len(tag_categories) == len(tag_category_nums)
        self.tag_categories = tag_categories
        self.tag_category_nums = tag_category_nums
        self.filename_tmpl = filename_tmpl
        self.num_categories = len(self.tag_categories)
        self.num_tags = sum(self.tag_category_nums)
        self.category2num = dict(zip(tag_categories, tag_category_nums))
        self.start_idx = [0]
        for i in range(self.num_categories - 1):
            self.start_idx.append(self.start_idx[-1] +
                                  self.tag_category_nums[i])
        self.category2startidx = dict(zip(tag_categories, self.start_idx))
        self.start_index = kwargs.pop('start_index', 0)
        self.dataset_type = None
        super().__init__(
            ann_file, pipeline, start_index=self.start_index, **kwargs)

    def load_annotations(self):
        """Load annotation file to get video information."""
        assert self.ann_file.endswith('.json')
        return self.load_json_annotations()

    def load_json_annotations(self):
        video_infos = mmcv.load(self.ann_file)
        num_videos = len(video_infos)

        video_info0 = video_infos[0]
        assert ('filename' in video_info0) != ('frame_dir' in video_info0)
        path_key = 'filename' if 'filename' in video_info0 else 'frame_dir'
        self.dataset_type = 'video' if path_key == 'filename' else 'rawframe'
        if self.dataset_type == 'rawframe':
            assert self.filename_tmpl is not None

        for i in range(num_videos):
            path_value = video_infos[i][path_key]
            if self.data_prefix is not None:
                path_value = osp.join(self.data_prefix, path_value)
            video_infos[i][path_key] = path_value

            # We will convert label to torch tensors in the pipeline
            video_infos[i]['categories'] = self.tag_categories
            video_infos[i]['category_nums'] = self.tag_category_nums
            if self.dataset_type == 'rawframe':
                video_infos[i]['filename_tmpl'] = self.filename_tmpl
                video_infos[i]['start_index'] = self.start_index
                video_infos[i]['modality'] = self.modality

        return video_infos

    @staticmethod
    def label2array(num, label):
        arr = np.zeros(num, dtype=np.float32)
        arr[label] = 1.
        return arr

    def evaluate(self,
                 results,
                 metric_options=dict(MeanAP=dict()),
                 logger=None):
        """Evaluation in HVU Video Dataset. We only support evaluating mAP for
        each tag categories. Since some tag categories are missing for some
        videos, we can not evaluate mAP for all tags.

        Args:
            results (list): Output results.
            metric_options (dict | None): Dict for metric options.
                Default: None.
            logger (logging.Logger | None): Logger for recording.
                Default: None.

        Returns:
            dict: Evaluation results dict.
        """
        metrics = list(metric_options)
        allowed_metrics = ('MeanAP', )

        if not isinstance(results, list):
            raise TypeError(f'results must be a list, but got {type(results)}')
        assert len(results) == len(self), (
            f'The length of results is not equal to the dataset len: '
            f'{len(results)} != {len(self)}')

        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        eval_results = {}
        metric_kwargs = {}
        gt_labels = [ann['label'] for ann in self.video_infos]

        for metric in metrics:

            if not hasattr(self, metric):
                metric_cfg = dict(
                    type=metric, logger=logger, **metric_options[metric])
                metric_func = build_metrics(metric_cfg)
                setattr(self, metric, metric_func)

            for category in self.tag_categories:

                metric_kwargs['category'] = category

                msg = f'Evaluating {metric} on {category} ...'
                msg = '\n' + msg if logger is None else msg
                print_log(msg, logger=logger)

                start_idx = self.category2startidx[category]
                num = self.category2num[category]

                preds = []
                for video_idx, result in enumerate(results):
                    if category in gt_labels[video_idx]:
                        preds.append(result[start_idx:start_idx + num])

                gts = []
                for gt_label in gt_labels:
                    if category in gt_label:
                        gts.append(gt_label[category])
                gts = [self.label2array(num, item) for item in gts]

                eval_res = getattr(self, metric)(preds, gts, metric_kwargs)
                eval_results.update(eval_res)

        return eval_results
