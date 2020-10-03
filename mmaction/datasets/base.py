import copy
import os.path as osp
from abc import ABCMeta, abstractmethod

import mmcv
import torch
from torch.utils.data import Dataset

from .pipelines import Compose


class BaseDataset(Dataset, metaclass=ABCMeta):
    """Base class for datasets.

    All datasets to process video should subclass it.
    All subclasses should overwrite:

    - Methods:`load_annotations`, supporting to load information from an
    annotation file.
    - Methods:`prepare_train_frames`, providing train data.
    - Methods:`prepare_test_frames`, providing test data.

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
        start_index (int): Specify a start index for frames in consideration of
            different filename format. However, when taking videos as input,
            it should be set to 0, since frames loaded from videos count
            from 0. Default: 1.
        modality (str): Modality of data. Support 'RGB', 'Flow'.
            Default: 'RGB'.
    """

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_prefix=None,
                 test_mode=False,
                 multi_class=False,
                 num_classes=None,
                 start_index=1,
                 modality='RGB'):
        super().__init__()

        self.ann_file = ann_file
        self.data_prefix = osp.realpath(data_prefix) if osp.isdir(
            data_prefix) else data_prefix
        self.test_mode = test_mode
        self.multi_class = multi_class
        self.num_classes = num_classes
        self.start_index = start_index
        self.modality = modality
        self.pipeline = Compose(pipeline)
        self.video_infos = self.load_annotations()

    @abstractmethod
    def load_annotations(self):
        """Load the annotation according to ann_file into video_infos."""
        pass

    # json annotations already looks like video_infos, so for each dataset,
    # this func should be the same
    def load_json_annotations(self):
        """Load json annotation file to get video information."""
        video_infos = mmcv.load(self.ann_file)
        num_videos = len(video_infos)
        path_key = 'frame_dir' if 'frame_dir' in video_infos[0] else 'filename'
        for i in range(num_videos):
            if self.data_prefix is not None:
                path_value = video_infos[i][path_key]
                path_value = osp.join(self.data_prefix, path_value)
                video_infos[i][path_key] = path_value
            if self.multi_class:
                assert self.num_classes is not None
                onehot = torch.zeros(self.num_classes)
                onehot[video_infos[i]['label']] = 1.
                video_infos[i]['label'] = onehot
            else:
                assert len(video_infos[i]['label']) == 1
                video_infos[i]['label'] = video_infos[i]['label'][0]
        return video_infos

    @abstractmethod
    def evaluate(self, results, metrics, logger):
        """Evaluation for the dataset.

        Args:
            results (list): Output results.
            metrics (str | sequence[str]): Metrics to be performed.
            logger (logging.Logger | None): Logger for recording.

        Returns:
            dict: Evaluation results dict.
        """
        pass

    def dump_results(self, results, out):
        """Dump data to json/yaml/pickle strings or files."""
        return mmcv.dump(results, out)

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

    def __len__(self):
        """Get the size of the dataset."""
        return len(self.video_infos)

    def __getitem__(self, idx):
        """Get the sample for either training or testing given index."""
        if self.test_mode:
            return self.prepare_test_frames(idx)
        else:
            return self.prepare_train_frames(idx)
