# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Callable, Dict, List, Optional, Union

from mmengine.utils import check_file_exist

from mmaction.registry import DATASETS
from .base import BaseActionDataset


@DATASETS.register_module()
class AudioDataset(BaseActionDataset):
    """Audio dataset for action recognition.

    The ann_file is a text file with multiple lines, and each line indicates
    a sample audio or extracted audio feature with the filepath, total frames
    of the raw video and label, which are split with a whitespace.
    Example of a annotation file:

    .. code-block:: txt
        some/directory-1.npy 163 1
        some/directory-2.npy 122 1
        some/directory-3.npy 258 2
        some/directory-4.npy 234 2
        some/directory-5.npy 295 3
        some/directory-6.npy 121 3

    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        data_prefix (dict): Path to a directory where
            audios are held. Defaults to ``dict(audio='')``.
        multi_class (bool): Determines whether it is a multi-class
            recognition dataset. Defaults to False.
        num_classes (int, optional): Number of classes in the dataset.
            Defaults to None.
    """

    def __init__(self,
                 ann_file: str,
                 pipeline: List[Union[Dict, Callable]],
                 data_prefix: Dict = dict(audio=''),
                 multi_class: bool = False,
                 num_classes: Optional[int] = None,
                 **kwargs) -> None:
        super().__init__(
            ann_file,
            pipeline,
            data_prefix=data_prefix,
            multi_class=multi_class,
            num_classes=num_classes,
            modality='Audio',
            **kwargs)

    def load_data_list(self) -> List[Dict]:
        """Load annotation file to get audio information."""
        check_file_exist(self.ann_file)
        data_list = []
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                line_split = line.strip().split()
                video_info = {}
                idx = 0
                filename = line_split[idx]
                if self.data_prefix['audio'] is not None:
                    filename = osp.join(self.data_prefix['audio'], filename)
                video_info['audio_path'] = filename
                idx += 1
                # idx for total_frames
                video_info['total_frames'] = int(line_split[idx])
                idx += 1
                # idx for label
                label = [int(x) for x in line_split[idx:]]
                assert label, f'missing label in line: {line}'
                if self.multi_class:
                    assert self.num_classes is not None
                    video_info['label'] = label
                else:
                    assert len(label) == 1
                    video_info['label'] = label[0]
                data_list.append(video_info)

        return data_list
