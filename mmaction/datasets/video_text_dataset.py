# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path as osp
from typing import Dict, List

from mmengine.fileio import exists

from mmaction.registry import DATASETS
from .base import BaseActionDataset


@DATASETS.register_module()
class VideoTextDataset(BaseActionDataset):
    """Video dataset for video-text task like video retrieval."""

    def load_data_list(self) -> List[Dict]:
        """Load annotation file to get video information."""
        exists(self.ann_file)
        data_list = []

        with open(self.ann_file) as f:
            video_dict = json.load(f)
            for filename, texts in video_dict.items():
                filename = osp.join(self.data_prefix['video'], filename)
                video_text_pairs = []
                for text in texts:
                    data_item = dict(filename=filename, text=text)
                    video_text_pairs.append(data_item)
                data_list.extend(video_text_pairs)

        return data_list
