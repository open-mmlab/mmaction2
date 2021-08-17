# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import torch

from .base import BaseDataset
from .builder import DATASETS


@DATASETS.register_module()
class AudioFeatureDataset(BaseDataset):
    """Audio feature dataset for video recognition. Reads the features
    extracted off-line. Annotation file can be that of the rawframe dataset,
    or:

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
        suffix (str): The suffix of the audio feature file. Default: '.npy'.
        kwargs (dict): Other keyword args for `BaseDataset`.
    """

    def __init__(self, ann_file, pipeline, suffix='.npy', **kwargs):
        self.suffix = suffix
        super().__init__(ann_file, pipeline, modality='Audio', **kwargs)

    def load_annotations(self):
        """Load annotation file to get video information."""
        if self.ann_file.endswith('.json'):
            return self.load_json_annotations()
        video_infos = []
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                line_split = line.strip().split()
                video_info = {}
                idx = 0
                filename = line_split[idx]
                if self.data_prefix is not None:
                    if not filename.endswith(self.suffix):
                        filename = osp.join(self.data_prefix,
                                            filename) + self.suffix
                    else:
                        filename = osp.join(self.data_prefix, filename)
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
                video_infos.append(video_info)

        return video_infos
