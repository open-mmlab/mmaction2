# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Callable, Dict, List, Optional, Union

import mmengine
from mmengine.logging import MMLogger

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
        valid_ratio (float, optional): The valid_ratio for videos in
            KineticsPose. For a video with n frames, it is a valid
            training sample only if n * valid_ratio frames have human
            pose. None means not applicable (only applicable to Kinetics
            Pose).Defaults to None.
        box_thr (float): The threshold for human proposals. Only boxes
            with confidence score larger than `box_thr` is kept. None
            means not applicable (only applicable to Kinetics). Allowed
            choices are 0.5, 0.6, 0.7, 0.8, 0.9. Defaults to 0.5.
    """

    def __init__(self,
                 ann_file: str,
                 pipeline: List[Union[Dict, Callable]],
                 split: Optional[str] = None,
                 valid_ratio: Optional[float] = None,
                 box_thr: float = 0.5,
                 **kwargs) -> None:
        self.split = split
        self.box_thr = box_thr
        assert box_thr in [.5, .6, .7, .8, .9]
        self.valid_ratio = valid_ratio

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

    def filter_data(self) -> List[Dict]:
        """Filter out invalid samples."""
        if self.valid_ratio is not None and isinstance(
                self.valid_ratio, float) and self.valid_ratio > 0:
            self.data_list = [
                x for x in self.data_list if x['valid'][self.box_thr] /
                x['total_frames'] >= self.valid_ratio
            ]
            for item in self.data_list:
                assert 'box_score' in item,\
                    'if valid_ratio is a positive number,' \
                    'item should have field `box_score`'
                anno_inds = (item['box_score'] >= self.box_thr)
                item['anno_inds'] = anno_inds

        logger = MMLogger.get_current_instance()
        logger.info(
            f'{len(self.data_list)} videos remain after valid thresholding')

        return self.data_list

    def get_data_info(self, idx: int) -> Dict:
        """Get annotation by index."""
        data_info = super().get_data_info(idx)

        # Sometimes we may need to load skeleton from the file
        if 'skeleton' in self.data_prefix:
            identifier = 'filename' if 'filename' in data_info \
                else 'frame_dir'
            ske_name = data_info[identifier]
            ske_path = osp.join(self.data_prefix['skeleton'],
                                ske_name + '.pkl')
            ske = mmengine.load(ske_path)
            for k in ske:
                data_info[k] = ske[k]

        return data_info
