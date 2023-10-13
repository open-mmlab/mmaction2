# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, List, Optional, Union

from mmengine.fileio import exists

from mmaction.registry import DATASETS
from mmaction.utils import ConfigType
from .base import BaseActionDataset


@DATASETS.register_module()
class ActionSegmentDataset(BaseActionDataset):

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
        """Load annotation file to get video information."""
        exists(self.ann_file)
        file_ptr = open(self.ann_file, 'r')  # read bundle
        list_of_examples = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        gts = [
            self.data_prefix['video'] + 'groundTruth/' + vid
            for vid in list_of_examples
        ]
        features_npy = [
            self.data_prefix['video'] + 'features/' + vid.split('.')[0] +
            '.npy' for vid in list_of_examples
        ]
        data_list = []

        file_ptr_d = open(self.data_prefix['video'] + '/mapping.txt', 'r')
        actions = file_ptr_d.read().split('\n')[:-1]
        file_ptr.close()
        actions_dict = dict()
        for a in actions:
            actions_dict[a.split()[1]] = int(a.split()[0])
        index2label = dict()
        for k, v in actions_dict.items():
            index2label[v] = k
        num_classes = len(actions_dict)

        # gts:txt list of examples:txt features_npy:npy
        for idx, feature in enumerate(features_npy):
            video_info = dict()
            feature_path = features_npy[idx]
            video_info['feature_path'] = feature_path
            video_info['actions_dict'] = actions_dict
            video_info['index2label'] = index2label
            video_info['ground_truth_path'] = gts[idx]
            video_info['num_classes'] = num_classes
            data_list.append(video_info)
        return data_list
