# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Callable, List, Optional, Union

import torch
from mmengine.utils import check_file_exist

from mmaction.registry import DATASETS
from mmaction.utils import ConfigType
from .base import BaseActionDataset


@DATASETS.register_module()
class AudioDataset(BaseActionDataset):
    """Audio dataset for action recognition. Annotation file can be that of the
    rawframe dataset, or:

    .. code-block:: txt
        some/directory-1.wav 163 1
        some/directory-2.wav 122 1
        some/directory-3.wav 258 2
        some/directory-4.wav 234 2
        some/directory-5.wav 295 3
        some/directory-6.wav 121 3

    .. code-block:: txt
        some/directory-1.npy 163 1
        some/directory-2.npy 122 1
        some/directory-3.npy 258 2
        some/directory-4.npy 234 2
        some/directory-5.npy 295 3
        some/directory-6.npy 121 3

    Args:
        ann_file (str): Path to the annotation file.
        pipeline (List[Union[dict, ConfigDict, Callable]]): A sequence of
            data transforms.
        data_prefix (dict or ConfigDict, optional): Path to a directory where
            audios are held. Defaults to ``dict(audio='')``.
        multi_class (bool): Determines whether it is a multi-class
            recognition dataset. Defaults to False.
        num_classes (int, optional): Number of classes in the dataset.
            Defaults to None.
        suffix (str): The suffix of the audio file. Defaults to ``.wav``.
    """

    def __init__(self,
                 ann_file: str,
                 pipeline: List[Union[ConfigType, Callable]],
                 data_prefix: ConfigType = dict(audio=''),
                 multi_class: bool = False,
                 num_classes: Optional[int] = None,
                 suffix: str = '.wav',
                 **kwargs) -> None:
        self.suffix = suffix
        super().__init__(
            ann_file,
            pipeline,
            data_prefix=data_prefix,
            multi_class=multi_class,
            num_classes=num_classes,
            modality='Audio',
            **kwargs)

    def load_data_list(self) -> List[dict]:
        """Load annotation file to get video information."""
        check_file_exist(self.ann_file)
        data_list = []
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                line_split = line.strip().split()
                video_info = {}
                idx = 0
                filename = line_split[idx]
                if self.data_prefix['audio'] is not None:
                    if not filename.endswith(self.suffix):
                        filename = osp.join(self.data_prefix['audio'],
                                            filename + self.suffix)
                    else:
                        filename = osp.join(self.data_prefix['audio'],
                                            filename)
                video_info['audio_path'] = filename
                idx += 1
                # idx for total_frames
                video_info['total_frames'] = int(line_split[idx])
                idx += 1
                # idx for label[s]
                label = [int(x) for x in line_split[idx:]]
                assert label, f'missing label in line: {line}'
                if self.multi_class:
                    assert self.num_classes is not None
                    onehot = torch.zeros(self.num_classes)
                    onehot[label] = 1.0
                    video_info['label'] = onehot
                else:
                    assert len(label) == 1
                    video_info['label'] = label[0]
                data_list.append(video_info)

        return data_list
