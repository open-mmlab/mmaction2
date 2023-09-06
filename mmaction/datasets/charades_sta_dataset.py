# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Callable, List, Optional, Union

import mmengine
import numpy as np
import torch
from mmengine.fileio import exists

from mmaction.registry import DATASETS
from mmaction.utils import ConfigType
from .base import BaseActionDataset

try:
    import nltk
    nltk_imported = True
except ImportError:
    nltk_imported = False


@DATASETS.register_module()
class CharadesSTADataset(BaseActionDataset):

    def __init__(self,
                 ann_file: str,
                 pipeline: List[Union[dict, Callable]],
                 word2id_file: str,
                 fps_file: str,
                 duration_file: str,
                 num_frames_file: str,
                 window_size: int,
                 ft_overlap: float,
                 data_prefix: Optional[ConfigType] = dict(video=''),
                 test_mode: bool = False,
                 **kwargs):
        if not nltk_imported:
            raise ImportError('nltk is required for CharadesSTADataset')

        self.fps_info = mmengine.load(fps_file)
        self.duration_info = mmengine.load(duration_file)
        self.num_frames = mmengine.load(num_frames_file)
        self.word2id = mmengine.load(word2id_file)
        self.ft_interval = int(window_size * (1 - ft_overlap))

        super().__init__(
            ann_file,
            pipeline=pipeline,
            data_prefix=data_prefix,
            test_mode=test_mode,
            **kwargs)

    def load_data_list(self) -> List[dict]:
        """Load annotation file to get video information."""
        exists(self.ann_file)
        data_list = []
        with open(self.ann_file) as f:
            anno_database = f.readlines()

        for item in anno_database:
            first_part, query_sentence = item.strip().split('##')
            query_sentence = query_sentence.replace('.', '')
            query_words = nltk.word_tokenize(query_sentence)
            query_tokens = [self.word2id[word] for word in query_words]
            query_length = len(query_tokens)
            query_tokens = torch.from_numpy(np.array(query_tokens))

            vid_name, start_time, end_time = first_part.split()
            duration = float(self.duration_info[vid_name])
            fps = float(self.fps_info[vid_name])

            gt_start_time = float(start_time)
            gt_end_time = float(end_time)

            gt_bbox = (gt_start_time / duration, min(gt_end_time / duration,
                                                     1))

            num_frames = int(self.num_frames[vid_name])
            proposal_frames = self.get_proposals(num_frames)

            proposals = proposal_frames / num_frames
            proposals = torch.from_numpy(proposals)
            proposal_indexes = proposal_frames / self.ft_interval
            proposal_indexes = proposal_indexes.astype(np.int32)

            info = dict(
                vid_name=vid_name,
                fps=fps,
                num_frames=num_frames,
                duration=duration,
                query_tokens=query_tokens,
                query_length=query_length,
                gt_start_time=gt_start_time,
                gt_end_time=gt_end_time,
                gt_bbox=gt_bbox,
                proposals=proposals,
                num_proposals=proposals.shape[0],
                proposal_indexes=proposal_indexes)
            data_list.append(info)
        return data_list

    def get_proposals(self, num_frames):
        proposals = (num_frames - 1) / 32 * np.arange(33)
        proposals = proposals.astype(np.int32)
        proposals = np.stack([proposals[:-1], proposals[1:]]).T
        return proposals

    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index."""
        data_info = super().get_data_info(idx)
        vid_name = data_info['vid_name']
        feature_path = os.path.join(self.data_prefix['video'],
                                    f'{vid_name}.pt')
        vid_feature = torch.load(feature_path)
        proposal_feats = []
        proposal_indexes = data_info['proposal_indexes'].clip(
            max=vid_feature.shape[0] - 1)
        for s, e in proposal_indexes:
            prop_feature, _ = vid_feature[s:e + 1].max(dim=0)
            proposal_feats.append(prop_feature)

        proposal_feats = torch.stack(proposal_feats)

        data_info['raw_feature'] = proposal_feats
        return data_info
