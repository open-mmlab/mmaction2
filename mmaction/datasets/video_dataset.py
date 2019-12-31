import copy
import os.path as osp

from torch.utils.data import Dataset

from .pipelines import Compose
from .registry import DATASETS


@DATASETS.register_module
class VideoDataset(Dataset):
    """A PyTorch video dataset for action recognition.

    This class is useful to load video data with multiple decode methods
    and applies pre-defined data pipeline with data augmentations (Normalize,
    MultiScaleCrop, Flip, etc.) and formatting operations (ToTensor, Collect,
    etc.) to return a dict with required values.

    Inputs:
        - ann_file (str): Path to an annotation file which store video info.
        - pipeline (list[dict | callable class]):
            A sequence of data augmentations and formatting operations.
        - data_prefix (str): Path to a directory where videos are held.
        - shorter_edge (int): shorter edge length of input videos.
        - input_size (int): width, height of input images
        - num_segments (int): number of extra frame segments
        - test_mode: store True when building test dataset.

    Annotation format:
        ['video_path' 'video_label'] format for each line
    """

    def __init__(self,
                 ann_file,
                 pipeline,
                 num_classes,
                 data_prefix=None,
                 shorter_edge=256,
                 input_size=224,
                 num_segments=1,
                 test_mode=False):
        self.ann_file = ann_file
        self.num_classes = num_classes
        self.data_prefix = data_prefix
        self.shorter_edge = shorter_edge
        self.input_size = input_size
        self.num_segments = num_segments
        self.test_mode = test_mode
        self.pipeline = Compose(pipeline)
        self.video_infos = self.load_annotations()

    def load_annotations(self):
        video_infos = []
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                filename, label = line.split(' ')
                filepath = osp.join(self.data_prefix, filename)
                video_infos.append(dict(filename=filepath, label=label))
        return video_infos

    def prepare_train_frames(self, idx):
        results = copy.deepcopy(self.video_infos[idx])
        print(results)
        return self.pipeline(results)

    def prepare_test_frames(self, idx):
        results = copy.deepcopy(self.video_infos[idx])
        return self.pipeline(results)

    def __len__(self):
        return len(self.video_infos)

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_frames(idx)
        else:
            return self.prepare_train_frames(idx)
