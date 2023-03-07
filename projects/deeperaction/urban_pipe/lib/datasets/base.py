# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta
from typing import Callable, List, Optional, Union

import torch
from mmengine.dataset import BaseDataset

from mmaction.utils import ConfigType


class BaseActionDataset(BaseDataset, metaclass=ABCMeta):
    """Base class for datasets.

    Args:
        ann_file (str): Path to the annotation file.
        pipeline (List[Union[dict, ConfigDict, Callable]]): A sequence of
            data transforms.
        data_prefix (dict or ConfigDict, optional): Path to a directory where
            videos are held. Defaults to None.
        test_mode (bool): Store True when building test or validation dataset.
            Defaults to False.
        multi_class (bool): Determines whether the dataset is a multi-class
            dataset. Defaults to False.
        num_classes (int, optional): Number of classes of the dataset, used in
            multi-class datasets. Defaults to None.
        start_index (int): Specify a start index for frames in consideration of
            different filename format. However, when taking videos as input,
            it should be set to 0, since frames loaded from videos count
            from 0. Defaults to 0.
        modality (str): Modality of data. Support ``RGB``, ``Flow``, ``Pose``,
            ``Audio``. Defaults to ``RGB``.
    """

    def __init__(self,
                 ann_file: str,
                 pipeline: List[Union[ConfigType, Callable]],
                 data_prefix: Optional[ConfigType] = dict(prefix=''),
                 test_mode: bool = False,
                 multi_class: bool = False,
                 num_classes: Optional[int] = None,
                 start_index: int = 0,
                 modality: str = 'RGB',
                 **kwargs) -> None:
        self.multi_class = multi_class
        self.num_classes = num_classes
        self.start_index = start_index
        self.modality = modality
        super().__init__(
            ann_file,
            pipeline=pipeline,
            data_prefix=data_prefix,
            test_mode=test_mode,
            **kwargs)

    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index."""
        data_info = super().get_data_info(idx)
        data_info['modality'] = self.modality
        data_info['start_index'] = self.start_index

        if self.multi_class:
            onehot = torch.zeros(self.num_classes)
            onehot[data_info['label']] = 1.
            data_info['label'] = onehot

        return data_info
