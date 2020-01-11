import copy
import os.path as osp

from .base import BaseDataset
from .registry import DATASETS


@DATASETS.register_module
class RawframeDataset(BaseDataset):
    """Rawframe dataset for action recognition.

    The dataset loads raw frames and apply specified transforms to return a
    dict containing the frame tensors and other information.

    The ann_file is a text file with multiple lines, and each line indicates
    sampled rawframes with the directory, total number and label, which
    are split with a whitespace. Example of a annotation file:

    ```
    some/path/000.mp4 frameNumber1 1
    some/path/001.mp4 frameNumber2 1
    some/path/002.mp4 frameNumber3 2
    some/path/003.mp4 frameNumber4 2
    some/path/004.mp4 frameNumber5 3
    some/path/005.mp4 frameNumber6 3
    ```

    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        data_prefix (str): Path to a directory where videos are held.
        input_size (int | tuple[int]): (width, height) of input images.
        test_mode (bool): store True when building test dataset.
        image_tmpl (str): Template for each filename.
    """

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_prefix=None,
                 test_mode=False,
                 image_tmpl='img_{0:0{width}}.jpg'):
        super(RawframeDataset, self).__init__(ann_file, pipeline, data_prefix,
                                              test_mode)
        self.image_tmpl = image_tmpl

    def load_annotations(self):
        video_infos = []
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                file_dir, total_frames, label = line.split(' ')
                file_dir = osp.join(self.data_prefix, file_dir)
                video_infos.append(
                    dict(
                        file_dir=file_dir,
                        total_frames=int(total_frames),
                        label=int(label)))
        return video_infos

    def prepare_train_frames(self, idx):
        results = copy.deepcopy(self.video_infos[idx])
        results['image_tmpl'] = self.image_tmpl
        return self.pipeline(results)

    def prepare_test_frames(self, idx):
        results = copy.deepcopy(self.video_infos[idx])
        results['image_tmpl'] = self.image_tmpl
        return self.pipeline(results)
