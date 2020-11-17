import copy
import os.path as osp
import pickle
from collections import defaultdict

from .base import BaseDataset
from .registry import DATASETS


@DATASETS.register_module()
class TubeDataset(BaseDataset):
    """Tube dataset for frame-level spatial temporal detection.

    Based on original "UCF101-24" and "JHMDB" annotation files, the dataset
    loads labels ('labels'), ground truth tubes ('gttubes'), frames number for
    each video ('nframes'), train video file list ('train_videos'), test video
    file list ('test_videos') and resolution for each video ('resolution'),
    and applies specified transformations to return a dict containing frame
    tensors and other information.

    Specifically, it can save arranged information into a pickle file to
    accelerate loading.

    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        preload_ann_file (str | None): Path to a pickle file, which has saved
            arranged information. Default: None.
        save_preload (bool): Whether to save the arranged information to a
            file. Default: False.
        num_classes (int): Number class of the dataset. Default: 24.
        data_prefix (str | None): Path to a directory where videos are held.
            Default: None.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
        encoding (str): Encode style to load pickle file.
            Default: 'iso-8859-1'.
        filename_tmpl (str): Template for each filename. Default: '{:05}.jpg'.
        split (int): Index of split to indicate the data part for train and
            test videos. Default: 1.
        tube_length (int): Length of tube to form the tubelet. Default: 7.
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
                 preload_ann_file=None,
                 save_preload=False,
                 num_classes=24,
                 data_prefix=None,
                 test_mode=False,
                 encoding='iso-8859-1',
                 filename_tmpl='{:05}.jpg',
                 split=1,
                 tube_length=7,
                 start_index=1,
                 modality='RGB'):
        self.preload_ann_file = preload_ann_file
        self.save_preload = save_preload
        self.filename_tmpl = filename_tmpl
        self.split = split
        self.tube_length = tube_length
        self.encoding = encoding
        super().__init__(
            ann_file,
            pipeline,
            data_prefix,
            test_mode=test_mode,
            num_classes=num_classes,
            start_index=start_index,
            modality=modality)

    @staticmethod
    def tubelet_in_tube(video_tube, frame_index, tube_length):
        return all([
            i in video_tube[:, 0]
            for i in range(frame_index, frame_index + tube_length)
        ])

    @staticmethod
    def tubelet_out_tube(video_tube, frame_index, tube_length):
        return all([
            i not in video_tube[:, 0]
            for i in range(frame_index, frame_index + tube_length)
        ])

    def check_tubelet(self, video_tubes, frame_index, tube_length):
        is_whole = all([
            self.tubelet_in_tube(video_tube, frame_index, tube_length)
            or self.tubelet_out_tube(video_tube, frame_index, tube_length)
            for video_tube in video_tubes
        ])
        has_gt = any([
            self.tubelet_in_tube(tube, frame_index, tube_length)
            for tube in video_tubes
        ])
        return is_whole and has_gt

    def load_annotations(self):
        pkl_data = pickle.load(
            open(self.ann_file, 'rb'), encoding=self.encoding)
        gt_tubes = pkl_data['gttubes']
        num_frames = pkl_data['nframes']
        train_videos = pkl_data['train_videos']
        test_videos = pkl_data['test_videos']
        resolution = pkl_data['resolution']

        self.gt_tubes = gt_tubes
        self.labels = pkl_data['labels']

        assert len(train_videos[self.split - 1]) + len(
            test_videos[self.split - 1]) == len(num_frames)
        videos = train_videos[
            self.split - 1] if not self.test_mode else test_videos[self.split -
                                                                   1]
        self.videos = videos

        if self.preload_ann_file is not None and osp.exists(
                self.preload_ann_file):
            video_infos = pickle.load(open(self.preload_ann_file, 'rb'))
        else:
            video_infos = []

            for video in videos:
                video_tubes = sum(gt_tubes[video].values(), [])
                for i in range(1, num_frames[video] + 2 - self.tube_length):
                    if self.check_tubelet(video_tubes, i, self.tube_length):
                        frame_dir = video
                        if self.data_prefix is not None:
                            frame_dir = osp.join(self.data_prefix, video)
                        gt_bboxes = defaultdict(list)

                        for label_index, tubes in gt_tubes[video].items():
                            for tube in tubes:
                                if i not in tube[:, 0]:
                                    continue
                                boxes = tube[
                                    (tube[:, 0] >= i) *
                                    (tube[:, 0] < i + self.tube_length), 1:5]
                                gt_bboxes[label_index].append(boxes)
                        video_info = {}
                        video_info['indice'] = (video, i)
                        video_info['video'] = video
                        video_info['frame_dir'] = frame_dir
                        video_info['total_frames'] = num_frames[video]
                        video_info['resolution'] = resolution[video]
                        video_info['gt_bboxes'] = gt_bboxes
                        video_infos.append(video_info)

            if self.save_preload:
                if self.preload_ann_file is None:
                    raise ValueError('preload annotation file should be '
                                     'assigned for saving')
                self.logger.info(f'Save tube info to {self.preload_ann_file}')
                pickle.dump(video_infos, open(self.preload_ann_file, 'wb'))

        return video_infos

    def prepare_train_frames(self, idx):
        results = copy.deepcopy(self.video_infos[idx])
        results['filename_tmpl'] = self.filename_tmpl
        results['modality'] = self.modality
        results['start_index'] = self.start_index
        results['tube_length'] = self.tube_length
        results['num_classes'] = self.num_classes

        return self.pipeline(results)

    def prepare_test_frames(self, idx):
        results = copy.deepcopy(self.video_infos[idx])
        results['filename_tmpl'] = self.filename_tmpl
        results['modality'] = self.modality
        results['start_index'] = self.start_index
        results['tube_length'] = self.tube_length
        results['num_classes'] = self.num_classes

        return self.pipeline(results)
