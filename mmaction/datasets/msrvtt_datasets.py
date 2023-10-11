# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path as osp
import re
from collections import Counter
from typing import Dict, List

from mmengine.fileio import exists

from mmaction.registry import DATASETS
from .base import BaseActionDataset


@DATASETS.register_module()
class MSRVTTVQA(BaseActionDataset):
    """MSR-VTT Video Question Answering dataset."""

    def load_data_list(self) -> List[Dict]:
        """Load annotation file to get video information."""
        exists(self.ann_file)
        data_list = []

        with open(self.ann_file) as f:
            data_lines = json.load(f)
            for data in data_lines:
                answers = data['answer']
                if isinstance(answers, str):
                    answers = [answers]
                count = Counter(answers)
                answer_weight = [i / len(answers) for i in count.values()]
                data_item = dict(
                    question_id=data['question_id'],
                    filename=osp.join(self.data_prefix['video'],
                                      data['video']),
                    question=pre_text(data['question']),
                    gt_answer=list(count.keys()),
                    gt_answer_weight=answer_weight)
                data_list.append(data_item)

        return data_list


@DATASETS.register_module()
class MSRVTTVQAMC(BaseActionDataset):
    """MSR-VTT VQA multiple choices dataset."""

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
class MSRVTTRetrieval(BaseActionDataset):
    """MSR-VTT Retrieval dataset."""

    def load_data_list(self) -> List[Dict]:
        """Load annotation file to get video information."""
        exists(self.ann_file)
        data_list = []

        with open(self.ann_file) as f:
            data_lines = json.load(f)
            video_idx = 0
            text_idx = 0
            for data in data_lines:
                # don't consider multiple videos or multiple captions
                video_path = osp.join(self.data_prefix['video'], data['video'])
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
