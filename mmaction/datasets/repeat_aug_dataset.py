# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from typing import Any, Callable, List, Optional, Sequence, Union

import numpy as np
from mmengine.dataset.utils import pseudo_collate
from mmengine.registry import Registry

from mmaction.registry import DATASETS
from mmaction.utils import ConfigType
from .video_dataset import VideoDataset

COLLATE_FUNCTIONS = Registry('Collate Functions')


def get_type(transform):
    if type(transform) == dict:
        return transform['type']
    elif callable(transform):
        return transform.__repr__().split('(')[0]


@DATASETS.register_module()
class RepeatAugDataset(VideoDataset):
    """Video dataset for action recognition.

    The dataset loads raw videos and apply specified transforms to return a
    dict containing the frame tensors and other information.

    The ann_file is a text file with multiple lines, and each line indicates
    a sample video with the filepath and label, which are split with a
    whitespace. Example of a annotation file:

    .. code-block:: txt

        some/path/000.mp4 1
        some/path/001.mp4 1
        some/path/002.mp4 2
        some/path/003.mp4 2
        some/path/004.mp4 3
        some/path/005.mp4 3


    Args:
        ann_file (str): Path to the annotation file.
        pipeline (List[Union[dict, ConfigDict, Callable]]): A sequence of
            data transforms.
        data_prefix (dict or ConfigDict): Path to a directory where videos
            are held. Defaults to ``dict(video='')``.
        multi_class (bool): Determines whether the dataset is a multi-class
            dataset. Defaults to False.
        num_classes (int, optional): Number of classes of the dataset, used in
            multi-class datasets. Defaults to None.
        start_index (int): Specify a start index for frames in consideration of
            different filename format. However, when taking videos as input,
            it should be set to 0, since frames loaded from videos count
            from 0. Defaults to 0.
        modality (str): Modality of data. Support ``RGB``, ``Flow``.
            Defaults to ``RGB``.
        test_mode (bool): Store True when building test or validation dataset.
            Defaults to False.
    """

    def __init__(self,
                 ann_file: str,
                 pipeline: List[Union[dict, Callable]],
                 data_prefix: ConfigType = dict(video=''),
                 num_repeats: int = 4,
                 multi_class: bool = False,
                 num_classes: Optional[int] = None,
                 start_index: int = 0,
                 modality: str = 'RGB',
                 **kwargs) -> None:

        flag = get_type(pipeline[0]) == 'DecordInit' and \
               get_type([2]['type']) == 'DecordDecode'

        assert flag, ('RepeatAugDataset requires decord as the video loading'
                      ' backend, will support more backends in the future')

        super().__init__(
            ann_file,
            pipeline=pipeline,
            data_prefix=data_prefix,
            multi_class=multi_class,
            num_classes=num_classes,
            start_index=start_index,
            modality=modality,
            test_mode=False,
            **kwargs)
        self.num_repeats = num_repeats

    def prepare_data(self, idx) -> List[dict]:
        """Get data processed by ``self.pipeline``.

        Reduce the video loading and decompressing.
        Args:
            idx (int): The index of ``data_info``.
        Returns:
            List[dict]: A list of length num_repeats.
        """
        transforms = self.pipeline.transforms

        data_info = self.get_data_info(idx)
        data_info = transforms[0](data_info)  # DecordInit

        frame_inds_list = []
        frame_inds_length = [0]
        for repeat in range(self.num_repeats):
            data_info_ = transforms[1](deepcopy(data_info))  # SampleFrames
            frame_inds = data_info_['frame_inds']
            frame_inds = frame_inds.reshape(-1)
            frame_inds_list.append(frame_inds)
            frame_inds_length.append(frame_inds.size + frame_inds_length[-1])

        data_info_['frame_inds'] = np.concatenate(frame_inds_list)
        data_info = transforms[2](data_info_)  # DecordDecode

        data_info_list = []
        for repeat in range(self.num_repeats):
            data_info_ = deepcopy(data_info)
            start = frame_inds_length[repeat]
            end = frame_inds_length[repeat + 1]
            data_info_['imgs'] = data_info_['imgs'][start:end]
            for transform in transforms[3:]:
                data_info_ = transform(data_info_)
            data_info_list.append(data_info_)
        return data_info_list


@COLLATE_FUNCTIONS.register_module()
def repeat_pseudo_collate(data_batch: Sequence) -> Any:
    data_batch = [i for j in data_batch for i in j]
    return pseudo_collate(data_batch)
