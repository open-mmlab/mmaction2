# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path as osp
import re
from collections import Counter, defaultdict
from typing import Callable, Dict, List, Optional, Union

from mmengine.fileio import exists

from mmaction.registry import DATASETS
from mmaction.utils import ConfigType
from .base import BaseActionDataset


@DATASETS.register_module()
class MSRVTT_RetMC(BaseActionDataset):
    """MSR-VTT Retrieval multiple choices dataset."""

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
        super().__init__(ann_file, pipeline, data_prefix, test_mode,
                         multi_class, num_classes, start_index, modality,
                         **kwargs)

    def load_data_list(self) -> List[Dict]:
        """Load annotation file to get video information."""
        exists(self.ann_file)
        data_list = []

        with open(self.ann_file) as f:
            data_lines = json.load(f)
            for data in data_lines:
                data_item = dict(
                    filename=osp.join(self.data_prefix['video'],
                                      data['video']),
                    label=data['answer'],
                    caption_options=[pre_text(c) for c in data['caption']])
                data_list.append(data_item)

        return data_list

@DATASETS.register_module()
class MSRVTT_Ret(BaseActionDataset):
    """MSR-VTT Retrieval dataset."""

    def __init__(self,
                 ann_file: str,
                 pipeline: List[Union[ConfigType, Callable]],
                 data_prefix: Optional[ConfigType] = dict(prefix=''),
                 test_mode: bool = False,
                 multi_class: bool = False,
                 num_classes: Optional[int] = None,
                 start_index: int = 0,
                 modality: str = 'RGB',
                 txt2multi_video: bool = False,
                 **kwargs) -> None:
        super().__init__(ann_file, pipeline, data_prefix, test_mode,
                         multi_class, num_classes, start_index, modality,
                         **kwargs)
        self.txt2multi_video = txt2multi_video

    def load_data_list(self) -> List[Dict]:
        """Load annotation file to get video information."""
        exists(self.ann_file)
        data_list = []

        with open(self.ann_file) as f:
            data_lines = json.load(f)
            video_idx = 0
            text_idx = 0
            for data in data_lines:
                # don't consider multiple videos or mulitple captions
                video_path = osp.join(self.data_prefix['video'],
                                        data['video'])
                data_item = dict(
                    filename=video_path,
                    text=[],
                    gt_video_id=[],
                    gt_text_id=[])
                if isinstance(data['caption'], str):
                    data['caption'] = [data['caption']]
                
                for text in data['caption']:
                    text = pre_text(text)
                    data_item['text'].append(text)
                    data_item['gt_video_id'].append(video_idx)
                    data_item['gt_text_id'].append(text_idx)
                    text_idx += 1

                video_idx += 1
                data_list.append(data_item)
        self.num_videos = video_idx
        self.num_texts = text_idx

        # debug_len = 1000
        # data_list = data_list[:debug_len]
        # self.num_videos = debug_len
        # self.num_texts = debug_len
        return data_list


def pre_text(text, max_l=None):
    text = re.sub(r"([,.'!?\"()*#:;~])", '', text.lower())
    text = text.replace('-', ' ').replace('/',
                                          ' ').replace('<person>', 'person')

    text = re.sub(r'\s{2,}', ' ', text)
    text = text.rstrip('\n').strip(' ')

    if max_l:  # truncate
        words = text.split(' ')
        if len(words) > max_l:
            text = ' '.join(words[:max_l])
    return text
