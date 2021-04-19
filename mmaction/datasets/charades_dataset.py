import copy
import os.path as osp
from collections import defaultdict

from .base import BaseDataset
from .registry import DATASETS


@DATASETS.register_module()
class CharadesDataset(BaseDataset):

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_prefix=None,
                 test_mode=False,
                 filename_tmpl_prefix='{}',
                 filename_tmpl_suffix='-{:06}.jpg',
                 multi_class=True,
                 num_classes=157,
                 start_index=1,
                 modality='RGB'):
        self.filename_tmpl_prefix = filename_tmpl_prefix
        self.filename_tmpl_suffix = filename_tmpl_suffix
        super().__init__(
            ann_file,
            pipeline,
            data_prefix,
            test_mode,
            multi_class=multi_class,
            num_classes=num_classes,
            start_index=start_index,
            modality=modality)

    @staticmethod
    def aggregate_labels(label_list):
        """Join a list of label list."""
        return list(set().union(*label_list))

    @staticmethod
    def load_image_lists(frame_list_file):
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
            if self.test_mode:
                video_info['label'] = self.aggregate_labels(labels[video_name])
            else:
                video_info['label'] = labels[video_name]
            if '{}' in self.filename_tmpl_prefix:
                video_info['filename_tmpl'] = self.filename_tmpl_prefix.format(
                    video_name)
            else:
                video_info['filename_tmpl'] = self.filename_tmpl_prefix
            video_info['filename_tmpl'] += self.filename_tmpl_suffix
            video_infos.append(video_info)

        return video_infos

    def prepare_train_frames(self, idx):
        """Prepare the frames for training given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        results['modality'] = self.modality
        results['start_index'] = self.start_index

        return self.pipeline(results)

    def prepare_test_frames(self, idx):
        """Prepare the frames for testing given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        results['modality'] = self.modality
        results['start_index'] = self.start_index

        return self.pipeline(results)
