# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path as osp
import re
from typing import Callable, Dict, List, Optional, Union

from mmengine.fileio import exists

from mmaction.registry import DATASETS
from mmaction.utils import ConfigType
from .base import BaseActionDataset


@DATASETS.register_module()
class MSRVTT_VQA(BaseActionDataset):
    """MSR-VTT Video Question Answering dataset."""

    def __init__(self,
                 ann_file: str,
                 pipeline: List[Union[ConfigType, Callable]],
                 k: int,
                 data_prefix: Optional[ConfigType] = dict(prefix=''),
                 test_mode: bool = False,
                 multi_class: bool = False,
                 num_classes: Optional[int] = None,
                 start_index: int = 0,
                 modality: str = 'RGB',
                 **kwargs) -> None:
        self.k = k
        super().__init__(ann_file, pipeline, data_prefix, test_mode,
                         multi_class, num_classes, start_index, modality,
                         **kwargs)

    def _get_answers_with_weights(self, raw_answers):
        if isinstance(raw_answers, str):
            raw_answers = [raw_answers]
        answer_weight = {}
        for answer in raw_answers:
            if answer in answer_weight.keys():
                answer_weight[answer] += 1 / len(raw_answers)
            else:
                answer_weight[answer] = 1 / len(raw_answers)

        answers = list(answer_weight.keys())
        weights = [answer_weight[a] for a in answers]
        answers = [answer + ' ' + self.eos for answer in answers]
        return answers, weights

    def load_data_list(self) -> List[Dict]:
        """Load annotation file to get video information."""
        exists(self.ann_file)
        data_list = []

        with open(self.ann_file) as f:
            data_lines = json.load(f)
            if not self.test_mode:
                answer_list = [d['answer'] for d in data_lines]
                answers, weights = self.get_answers_with_weights(answer_list)
                for data, answer, weight in zip(data_lines, answers, weights):
                    data_item = dict(
                        question_id=data['question_id'],
                        filename=osp.join(self.data_prefix['video'],
                                          data['video']),
                        question=pre_text(data['question']),
                        gt_answer=answer,
                        weight=weight,
                        k=self.k)
                    data_list.append(data_item)
            else:
                for data in data_lines:
                    data_item = dict(
                        question_id=data['question_id'],
                        filename=osp.join(self.data_prefix['video'],
                                          data['video']),
                        question=pre_text(data['question']),
                        gt_answer=data['answer'],
                        k=self.k)
                    data_list.append(data_item)
        # data_list = data_list[:100]
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
