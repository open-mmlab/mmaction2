# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, List, Optional, Union

import mmengine
from mmengine.utils import check_file_exist

from mmaction.registry import DATASETS
from mmaction.utils import ConfigType
from .base import BaseActionDataset


@DATASETS.register_module()
class ActivityNetDataset(BaseActionDataset):
    """ActivityNet dataset for temporal action localization. The dataset loads
    raw features and apply specified transforms to return a dict containing the
    frame tensors and other information. The ann_file is a json file with
    multiple objects, and each object has a key of the name of a video, and
    value of total frames of the video, total seconds of the video, annotations
    of a video, feature frames (frames covered by features) of the video, fps
    and rfps. Example of a annotation file:

    .. code-block:: JSON
        {
            "v_--1DO2V4K74":  {
                "duration_second": 211.53,
                "duration_frame": 6337,
                "annotations": [
                    {
                        "segment": [
                            30.025882995319815,
                            205.2318595943838
                        ],
                        "label": "Rock climbing"
                    }
                ],
                "feature_frame": 6336,
                "fps": 30.0,
                "rfps": 29.9579255898
            },
            "v_--6bJUbfpnQ": {
                "duration_second": 26.75,
                "duration_frame": 647,
                "annotations": [
                    {
                        "segment": [
                            2.578755070202808,
                            24.914101404056165
                        ],
                        "label": "Drinking beer"
                    }
                ],
                "feature_frame": 624,
                "fps": 24.0,
                "rfps": 24.1869158879
            },
            ...
        }
    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        data_prefix (dict or ConfigDict): Path to a directory where videos are
            held. Defaults to ``dict(video='')``.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
    """

    def __init__(self,
                 ann_file: str,
                 pipeline: List[Union[dict, Callable]],
                 data_prefix: Optional[ConfigType] = dict(video=''),
                 test_mode: bool = False,
                 **kwargs):

        super().__init__(
            ann_file,
            pipeline=pipeline,
            data_prefix=data_prefix,
            test_mode=test_mode,
            **kwargs)

    def load_data_list(self) -> List[dict]:
        check_file_exist(self.ann_file)
        data_list = []
        anno_database = mmengine.load(self.ann_file)
        for video_name in anno_database:
            video_info = anno_database[video_name]
            feature_path = video_name + '.csv'
            feature_path = '%s/%s' % (self.data_prefix['video'], feature_path)
            video_info['feature_path'] = feature_path
            video_info['video_name'] = video_name
            data_list.append(video_info)
        return data_list
