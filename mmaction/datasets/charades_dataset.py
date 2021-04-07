import copy
import os.path as osp
from collections import defaultdict

import numpy as np

from .base import BaseDataset
from .registry import DATASETS


@DATASETS.register_module()
class CharadesDataset(BaseDataset):

    def __init__(self, filename_tmpl='{}-{}.jpg', *args, **kwargs):
        self.filename_tmpl = filename_tmpl
        super().__init__(*args, **kwargs)

    @staticmethod
    def load_image_lists(frame_list_file, return_list=False):
        image_paths = defaultdict(list)
        labels = defaultdict(list)
        with open(frame_list_file, 'r') as f:
            assert f.readline().startswith('original_video_id')
            for line in f.readlines():
                row = line.split()
                # original_vido_id video_id frame_id path labels
                assert len(row) == 5
                video_name = row[0]
                path = row[3]
                image_paths[video_name].append(path)
                frame_labels = row[-1].replace('"', '')
                if frame_labels != '':
                    labels[video_name].append(
                        [int(x) for x in frame_labels.split(',')])
                else:
                    labels[video_name].append([])

        if return_list:
            image_paths = list(image_paths.values())
            labels = list(labels.values())
            return image_paths, labels
        return dict(image_paths), dict(labels)

    def load_annotations(self):
        """Load annotation file to get video information."""
        video_infos = []
        image_paths, labels = self.load_image_lists(self.ann_file)
        for video_name in image_paths:
            video_info = {}
            frame_dir = osp.dirname(image_paths[video_name][0])
            if self.data_prefix is not None:
                frame_dir = osp.join(self.data_prefix, frame_dir)
            video_info['frame_dir'] = frame_dir
            video_info['total_frames'] = len(image_paths[video_name])
            video_info['labels'] = labels[video_name]
            video_info['filename_tmpl'] = self.filename_tmpl.format(
                video_name, '{:06}')
            video_infos.append(video_info)

        return video_infos

    def prepare_train_frames(self, idx):
        """Prepare the frames for training given the index."""
        if self.sample_by_class:
            # Then, the idx is the class index
            samples = self.video_infos_by_class[idx]
            results = copy.deepcopy(np.random.choice(samples))
        else:
            results = copy.deepcopy(self.video_infos[idx])
        results['modality'] = self.modality
        results['start_index'] = self.start_index

        return self.pipeline(results)

    def prepare_test_frames(self, idx):
        """Prepare the frames for testing given the index."""
        if self.sample_by_class:
            # Then, the idx is the class index
            samples = self.video_infos_by_class[idx]
            results = copy.deepcopy(np.random.choice(samples))
        else:
            results = copy.deepcopy(self.video_infos[idx])
        results['modality'] = self.modality
        results['start_index'] = self.start_index

        return self.pipeline(results)
