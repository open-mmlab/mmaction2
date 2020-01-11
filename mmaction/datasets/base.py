from abc import ABCMeta, abstractmethod

from torch.utils.data import Dataset

from .pipelines import Compose


class BaseDataset(Dataset):
    """Base class for VideoDatasets and rawframeDatasets.

    All datasets to process video should subclass it.
    All subclasses should overwrite:
        Methods:`load_annotations`, supporting to load information
            from an annotation file.
        Methods:`prepare_train_frames`, providing train data.
        Methods:`prepare_test_frames`, providing test data.
    """

    __metaclass__ = ABCMeta

    def __init__(self, ann_file, pipeline, data_prefix, test_mode):
        super(BaseDataset, self).__init__()

        self.ann_file = ann_file
        self.data_prefix = data_prefix
        self.test_mode = test_mode
        self.pipeline = Compose(pipeline)
        self.video_infos = self.load_annotations()

    @abstractmethod
    def load_annotations(self):
        pass

    @abstractmethod
    def prepare_train_frames(self, idx):
        pass

    @abstractmethod
    def prepare_test_frames(self, idx):
        pass

    def __len__(self):
        return len(self.video_infos)

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_frames(idx)
        else:
            return self.prepare_train_frames(idx)
