# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import numpy as np

from ..utils import get_root_logger
from .base import BaseDataset
from .builder import DATASETS


@DATASETS.register_module()
class PoseDataset(BaseDataset):
    """Pose dataset for action recognition.

    The dataset loads pose and apply specified transforms to return a
    dict containing pose information.

    The ann_file is a pickle file, the json file contains a list of
    annotations, the fields of an annotation include frame_dir(video_id),
    total_frames, label, kp, kpscore.

    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        split (str | None): The dataset split used. Only applicable to UCF or
            HMDB. Allowed choiced are 'train1', 'test1', 'train2', 'test2',
            'train3', 'test3'. Default: None.
        valid_ratio (float | None): The valid_ratio for videos in KineticsPose.
            For a video with n frames, it is a valid training sample only if
            n * valid_ratio frames have human pose. None means not applicable
            (only applicable to Kinetics Pose). Default: None.
        box_thr (str | None): The threshold for human proposals. Only boxes
            with confidence score larger than `box_thr` is kept. None means
            not applicable (only applicable to Kinetics Pose [ours]). Allowed
            choices are '0.5', '0.6', '0.7', '0.8', '0.9'. Default: None.
        class_prob (dict | None): The per class sampling probability. If not
            None, it will override the class_prob calculated in
            BaseDataset.__init__(). Default: None.
        **kwargs: Keyword arguments for ``BaseDataset``.
    """

    def __init__(self,
                 ann_file,
                 pipeline,
                 split=None,
                 valid_ratio=None,
                 box_thr=None,
                 class_prob=None,
                 **kwargs):
        modality = 'Pose'
        # split, applicable to ucf or hmdb
        self.split = split

        super().__init__(
            ann_file, pipeline, start_index=0, modality=modality, **kwargs)

        # box_thr, which should be a string
        self.box_thr = box_thr
        if self.box_thr is not None:
            assert box_thr in ['0.5', '0.6', '0.7', '0.8', '0.9']

        # Thresholding Training Examples
        self.valid_ratio = valid_ratio
        if self.valid_ratio is not None:
            assert isinstance(self.valid_ratio, float)
            if self.box_thr is None:
                self.video_infos = self.video_infos = [
                    x for x in self.video_infos
                    if x['valid_frames'] / x['total_frames'] >= valid_ratio
                ]
            else:
                key = f'valid@{self.box_thr}'
                self.video_infos = [
                    x for x in self.video_infos
                    if x[key] / x['total_frames'] >= valid_ratio
                ]
                if self.box_thr != '0.5':
                    box_thr = float(self.box_thr)
                    for item in self.video_infos:
                        inds = [
                            i for i, score in enumerate(item['box_score'])
                            if score >= box_thr
                        ]
                        item['anno_inds'] = np.array(inds)

        if class_prob is not None:
            self.class_prob = class_prob

        logger = get_root_logger()
        logger.info(f'{len(self)} videos remain after valid thresholding')

    def load_annotations(self):
        """Load annotation file to get video information."""
        assert self.ann_file.endswith('.pkl')
        return self.load_pkl_annotations()

    def load_pkl_annotations(self):
        data = mmcv.load(self.ann_file)

        if self.split:
            split, data = data['split'], data['annotations']
            identifier = 'filename' if 'filename' in data[0] else 'frame_dir'
            data = [x for x in data if x[identifier] in split[self.split]]

        for item in data:
            # Sometimes we may need to load anno from the file
            if 'filename' in item:
                item['filename'] = osp.join(self.data_prefix, item['filename'])
            if 'frame_dir' in item:
                item['frame_dir'] = osp.join(self.data_prefix,
                                             item['frame_dir'])
        return data
