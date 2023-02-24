# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Callable, Dict, List, Optional, Union

import mmengine

from mmaction.registry import DATASETS
from .base import BaseActionDataset


@DATASETS.register_module()
class PoseDataset(BaseActionDataset):
    """Pose dataset for action recognition.

    The dataset loads pose and apply specified transforms to return a
    dict containing pose information.

    The ann_file is a pickle file, the json file contains a list of
    annotations, the fields of an annotation include frame_dir(video_id),
    total_frames, label, kp, kpscore.

    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        split (str, optional): The dataset split used. For UCF101 and
            HMDB51, allowed choices are 'train1', 'test1', 'train2',
            'test2', 'train3', 'test3'. For NTURGB+D, allowed choices
            are 'xsub_train', 'xsub_val', 'xview_train', 'xview_val'.
            For NTURGB+D 120, allowed choices are 'xsub_train',
            'xsub_val', 'xset_train', 'xset_val'. For FineGYM,
            allowed choices are 'train', 'val'. Defaults to None.
    """

    def __init__(self,
                 ann_file: str,
                 pipeline: List[Union[Dict, Callable]],
                 split: Optional[str] = None,
                 **kwargs) -> None:
        self.split = split
        super().__init__(
            ann_file, pipeline=pipeline, modality='Pose', **kwargs)

    def load_data_list(self) -> List[Dict]:
        """Load annotation file to get skeleton information."""
        assert self.ann_file.endswith('.pkl')
        mmengine.exists(self.ann_file)
        data_list = mmengine.load(self.ann_file)

        if self.split is not None:
            split, annos = data_list['split'], data_list['annotations']
            identifier = 'filename' if 'filename' in annos[0] else 'frame_dir'
            split = set(split[self.split])
            data_list = [x for x in annos if x[identifier] in split]

        # Sometimes we may need to load video from the file
        if 'video' in self.data_prefix:
            for item in data_list:
                if 'filename' in item:
                    item['filename'] = osp.join(self.data_prefix['video'],
                                                item['filename'])
                if 'frame_dir' in item:
                    item['frame_dir'] = osp.join(self.data_prefix['video'],
                                                 item['frame_dir'])
        return data_list
