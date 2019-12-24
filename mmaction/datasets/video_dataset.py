import copy
import os.path as osp

from torch.utils.data import Dataset


class VideoDataset(Dataset):

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_prefix=None,
                 shorter_edge=256,
                 input_size=224,
                 num_segments=1,
                 test_mode=False):
        self.ann_file = ann_file
        self.data_prefix = data_prefix

    def load_annotations(self, ann_file):
        video_infos = []
        with open(ann_file, 'r') as fin:
            for line in fin:
                filename, label = line.split(' ')
                filepath = osp.join(self.data_prefix, filename)
                video_infos.append(dict(filename=filepath, label=label))
        self.video_infos = video_infos

    def prepare_train_frames(self, idx):
        results = copy.deepcopy(self.video_infos[idx])
        return self.pipeline(results)

    def prepare_test_frames(self, idx):
        pass

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_frames()
        else:
            return self.prepare_train_frames()
