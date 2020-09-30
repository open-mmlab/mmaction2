import copy
import os.path as osp

import torch
from mmcv.utils import print_log

from ..core import mean_average_precision, mean_class_accuracy, top_k_accuracy
from .base import BaseDataset
from .registry import DATASETS


@DATASETS.register_module()
class ImageDataset(BaseDataset):
    """Image dataset for action recognition.

    The dataset loads image list and apply specified transforms to return a
    dict containing the image tensors and other information.

    The ann_file is a text file with multiple lines, and each line indicates
    the image path and the image label, which are split with a whitespace.
    Example of a annotation file:

    .. code-block:: txt

        path/to/image1.jpg 1
        path/to/image2.jpg 1
        path/to/image3.jpg 2
        path/to/image4.jpg 2
        path/to/image5.jpg 3
        path/to/image6.jpg 3

    Example of a multi-class annotation file:

    .. code-block:: txt

        path/to/image1.jpg 1 3 5
        path/to/image2.jpg 1 2
        path/to/image3.jpg 2
        path/to/image4.jpg 2 4 6 8
        path/to/image5.jpg 3
        path/to/image6.jpg 3

    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        data_prefix (str): Path to a directory where videos are held.
            Default: None.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
        multi_class (bool): Determines whether it is a multi-class
            recognition dataset. Default: False.
        num_classes (int): Number of classes in the dataset. Default: None.
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
                 modality='RGB'):

        super().__init__(
            ann_file,
            pipeline,
            data_prefix,
            test_mode,
            multi_class,
            num_classes,
            modality=modality)

    def load_annotations(self):
        """Load annotation file to get video information."""
        if self.ann_file.endswith('.json'):
            return self.load_json_annotations()
        video_infos = []
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                line_split = line.strip().split()
                video_info = {}
                idx = 0
                # idx for frame_dir
                frame_dir = line_split[idx]
                if self.data_prefix is not None:
                    frame_dir = osp.join(self.data_prefix, frame_dir)
                video_info['frame_dir'] = frame_dir
                idx += 1
                if self.with_offset:
                    # idx for offset and total_frames
                    video_info['offset'] = int(line_split[idx])
                    video_info['total_frames'] = int(line_split[idx + 1])
                    idx += 2
                else:
                    # idx for total_frames
                    video_info['total_frames'] = int(line_split[idx])
                    idx += 1
                # idx for label[s]
                label = [int(x) for x in line_split[idx:]]
                assert len(label), f'missing label in line: {line}'
                if self.multi_class:
                    assert self.num_classes is not None
                    onehot = torch.zeros(self.num_classes)
                    onehot[label] = 1.0
                    video_info['label'] = onehot
                else:
                    assert len(label) == 1
                    video_info['label'] = label[0]
                video_infos.append(video_info)

        return video_infos

    def prepare_train_frames(self, idx):
        """Prepare the frames for training given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        results['filename_tmpl'] = self.filename_tmpl
        results['modality'] = self.modality
        results['start_index'] = self.start_index
        return self.pipeline(results)

    def prepare_test_frames(self, idx):
        """Prepare the frames for testing given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        results['filename_tmpl'] = self.filename_tmpl
        results['modality'] = self.modality
        results['start_index'] = self.start_index
        return self.pipeline(results)

    def evaluate(self,
                 results,
                 metrics='top_k_accuracy',
                 topk=(1, 5),
                 logger=None):
        """Evaluation in rawframe dataset.

        Args:
            results (list): Output results.
            metrics (str | sequence[str]): Metrics to be performed.
                Defaults: 'top_k_accuracy'.
            logger (obj): Training logger. Defaults: None.
            topk (int | tuple[int]): K value for top_k_accuracy metric.
                Defaults: (1, 5).
            logger (logging.Logger | None): Logger for recording.
                Default: None.

        Returns:
            dict: Evaluation results dict.
        """

        if not isinstance(results, list):
            raise TypeError(f'results must be a list, but got {type(results)}')
        assert len(results) == len(self), (
            f'The length of results is not equal to the dataset len: '
            f'{len(results)} != {len(self)}')

        if not isinstance(topk, (int, tuple)):
            raise TypeError(
                f'topk must be int or tuple of int, but got {type(topk)}')

        if isinstance(topk, int):
            topk = (topk, )

        metrics = metrics if isinstance(metrics, (list, tuple)) else [metrics]
        allowed_metrics = [
            'top_k_accuracy', 'mean_class_accuracy', 'mean_average_precision'
        ]
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        eval_results = {}
        gt_labels = [ann['label'] for ann in self.video_infos]

        for metric in metrics:
            msg = f'Evaluating {metric}...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric == 'top_k_accuracy':
                top_k_acc = top_k_accuracy(results, gt_labels, topk)
                log_msg = []
                for k, acc in zip(topk, top_k_acc):
                    eval_results[f'top{k}_acc'] = acc
                    log_msg.append(f'\ntop{k}_acc\t{acc:.4f}')
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                continue

            if metric == 'mean_class_accuracy':
                mean_acc = mean_class_accuracy(results, gt_labels)
                eval_results['mean_class_accuracy'] = mean_acc
                log_msg = f'\nmean_acc\t{mean_acc:.4f}'
                print_log(log_msg, logger=logger)
                continue

            if metric == 'mean_average_precision':
                gt_labels = [label.cpu().numpy() for label in gt_labels]
                mAP = mean_average_precision(results, gt_labels)
                eval_results['mean_average_precision'] = mAP
                log_msg = f'\nmean_average_precision\t{mAP:.4f}'
                print_log(log_msg, logger=logger)
                continue

        return eval_results
