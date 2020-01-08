import copy
import os.path as osp

from torch.utils.data import Dataset

from .pipelines import Compose
from .registry import DATASETS


@DATASETS.register_module
class VideoDataset(Dataset):
    """Video dataset for action recognition.

    The dataset loads raw videos and apply specified transforms to return a
    dict containing the frame tensors and other infomation.

    The ann_file is a text file with multiple lines, and each line indicate a
    sample video with the filepath and label, which are split with a
    whitespace. Example of a annotation file:

    ```
    some/path/000.mp4 1
    some/path/001.mp4 1
    some/path/002.mp4 2
    some/path/003.mp4 2
    some/path/004.mp4 3
    some/path/005.mp4 3
    ```

    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        data_prefix (str): Path to a directory where videos are held.
        input_size (int | tuple[int]): (width, height) of input images.
        test_mode (bool): store True when building test dataset.
    """

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_prefix=None,
                 input_size=224,
                 test_mode=False):
        self.ann_file = ann_file
        self.data_prefix = data_prefix
        self.input_size = input_size
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
