# Copyright (c) OpenMMLab. All rights reserved.
import torch
import os.path as osp

from mmengine.dataset import BaseDataset
from mmengine.utils import check_file_exist

from mmaction.registry import DATASETS


@DATASETS.register_module()
class VideoDataset(BaseDataset):
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
        pipeline (list[dict | callable]): A sequence of data transforms.
        data_prefix (dict): Path to a directory where video are held. Default: None.
        start_index (int): Specify a start index for frames in consideration of
            different filename format. However, when taking videos as input,
            it should be set to 0, since frames loaded from videos count
            from 0. Default: 0.
        multi_class (bool): Determines whether the dataset is a multi-class
            dataset. Default: False.
        num_classes (int | None): Number of classes of the dataset, used in
            multi-class datasets. Default: None.
        modality (str): Modality of data. Support 'RGB', 'Flow', 'Audio'.
            Default: 'RGB'.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
        **kwargs: Keyword arguments for ``BaseDataset``.
    """

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_prefix=dict(video=None),
                 multi_class=False,
                 num_classes=None,
                 start_index=0,
                 modality='RGB',
                 test_mode=False,
                 **kwargs):
        self.multi_class = multi_class
        self.num_classes = num_classes
        self.start_index = start_index
        self.modality = modality
        super().__init__(ann_file,
                         pipeline=pipeline,
                         data_prefix=data_prefix,
                         test_mode=test_mode,
                         **kwargs)

    def load_data_list(self):
        """Load annotation file to get video information."""
        check_file_exist(self.ann_file)
        data_list = []
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                line_split = line.strip().split()
                if self.multi_class:
                    assert self.num_classes is not None
                    filename, label = line_split[0], line_split[1:]
                    label = list(map(int, label))
                else:
                    filename, label = line_split
                    label = int(label)
                if self.data_prefix['video'] is not None:
                    filename = osp.join(self.data_prefix['video'], filename)
                data_list.append(dict(filename=filename, label=label))
        return data_list

    def get_data_info(self, idx: int) -> dict:
        data_info = super().get_data_info(idx)
        data_info['modality'] = self.modality
        data_info['start_index'] = self.start_index

        if self.multi_class:
            onehot = torch.zeros(self.num_classes)
            onehot[data_info['label']] = 1.
            data_info['label'] = onehot

        return data_info
