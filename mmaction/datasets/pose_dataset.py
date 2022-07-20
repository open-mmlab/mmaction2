# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, List, Optional, Union

from mmengine.dataset import BaseDataset
from mmengine.fileio import load
from mmengine.utils import check_file_exist

from mmaction.registry import DATASETS
from mmaction.utils import ConfigType


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
        pipeline (list): A sequence of data transforms.
        split (str, optional): The dataset split used. Only applicable to
            ``UCF`` or ``HMDB``. Allowed choices are ``train1``, ``test1``,
            ``train2``, ``test2``, ``train3``, ``test3``. Defaults to None.
        start_index (int): Specify a start index for frames in consideration of
            different filename format. Defaults to 1.
        modality (str): Modality of data. Support ``RGB``, ``Flow``.
            Defaults to ``Pose``.
    """

    def __init__(self,
                 ann_file: str,
                 pipeline: List[Union[ConfigType, Callable]],
                 split: Optional[str] = None,
                 start_index: int = 0,
                 modality: str = 'Pose',
                 **kwargs) -> None:
        # split, applicable to ucf or hmdb
        self.split = split
        self.start_index = start_index
        self.modality = modality
        super().__init__(ann_file, pipeline=pipeline, **kwargs)

    def load_data_list(self) -> List[dict]:
        """Load annotation file to get skeleton information."""
        assert self.ann_file.endswith('.pkl')
        check_file_exist(self.ann_file)
        data_list = load(self.ann_file)

        if self.split is not None:
            split, data = data_list['split'], data_list['annotations']
            identifier = 'filename' if 'filename' in data[0] else 'frame_dir'
            data_list = [x for x in data if x[identifier] in split[self.split]]

        return data_list

    def get_data_info(self, idx: int) -> dict:
        data_info = super().get_data_info(idx)
        data_info['modality'] = self.modality
        data_info['start_index'] = self.start_index

        return data_info
