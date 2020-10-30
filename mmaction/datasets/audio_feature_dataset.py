import os.path as osp

import torch
from mmcv.utils import print_log

from ..core import mean_class_accuracy, top_k_accuracy
from .base import BaseDataset
from .registry import DATASETS


@DATASETS.register_module()
class AudioFeatureDataset(BaseDataset):
    """Audio feature dataset for video recognition. Reads the features
    extracted off-line. Annotation file can be that of the rawframe dataset,
    or:

    .. code-block:: txt

        some/directory-1.npy 163 1
        some/directory-2.npy 122 1
        some/directory-3.npy 258 2
        some/directory-4.npy 234 2
        some/directory-5.npy 295 3
        some/directory-6.npy 121 3

    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        suffix (str): The suffix of the audio feature file. Default: '.npy'.
        kwargs (dict): Other keyword args for `BaseDataset`.
    """

    def __init__(self, ann_file, pipeline, suffix='.npy', **kwargs):
        self.suffix = suffix
        super().__init__(ann_file, pipeline, modality='Audio', **kwargs)

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
                filename = line_split[idx]
                if self.data_prefix is not None:
                    if not filename.endswith(self.suffix):
                        filename = osp.join(self.data_prefix,
                                            filename) + self.suffix
                    else:
                        filename = osp.join(self.data_prefix, filename)
                video_info['audio_path'] = filename
                idx += 1
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

    def evaluate(self,
                 results,
                 metrics='top_k_accuracy',
                 topk=(1, 5),
                 logger=None):
        """Evaluation in audio feature dataset.

        Args:
            results (list): Output results.
            metrics (str | sequence[str]): Metrics to be performed.
                Defaults: 'top_k_accuracy'.
            logger (obj): Training logger. Defaults: None.
            topk (tuple[int]): K value for top_k_accuracy metric.
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

        metrics = metrics if isinstance(metrics, (list, tuple)) else [metrics]
        allowed_metrics = ['top_k_accuracy', 'mean_class_accuracy']
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

        return eval_results
