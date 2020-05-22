import copy
import os.path as osp
from abc import ABCMeta, abstractmethod

import mmcv
from torch.utils.data import Dataset

from .pipelines import Compose


class BaseDataset(Dataset, metaclass=ABCMeta):
    """Base class for datasets.

    All datasets to process video should subclass it.
    All subclasses should overwrite:
        Methods:`load_annotations`, supporting to load information
            from an annotation file.
        Methods:`prepare_train_frames`, providing train data.
        Methods:`prepare_test_frames`, providing test data.

    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        data_prefix (str): Path to a directory where videos are held.
            Default: None.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
        multi_class (bool): Determines whether the dataset is a multi-class
            dataset. Default: False.
        num_classes (int): Number of classes of the dataset, used in
            multi-class datasets. Default: None.
    """

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_prefix=None,
                 test_mode=False,
                 multi_class=False,
                 num_classes=None):
        super(BaseDataset, self).__init__()

        self.ann_file = ann_file
        self.data_prefix = osp.realpath(data_prefix) if osp.isdir(
            data_prefix) else data_prefix
        self.test_mode = test_mode
        self.multi_class = multi_class
        self.num_classes = num_classes
        self.pipeline = Compose(pipeline)
        self.video_infos = self.load_annotations()

    @abstractmethod
    def load_annotations(self):
        pass

    @abstractmethod
    def evaluate(self, results, metrics, logger):
        pass

    def dump_results(self, results, out):
        return mmcv.dump(results, out)

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
