import copy
import json
import os
import os.path as osp
import warnings

import mmcv
import numpy as np

from ..core import average_recall_at_avg_proposals, pairwise_temporal_iou
from .base import BaseDataset
from .registry import DATASETS


@DATASETS.register_module()
class ActivityNetDataset(BaseDataset):
    """ActivityNet dataset for temporal action localization.

    The dataset loads raw features and apply specified transforms to return a
    dict containing the frame tensors and other information.

    The ann_file is a json file with multiple objects, and each object has a
    key of the name of a video, and value of total frames of the video, total
    seconds of the video, annotations of a video, feature frames (frames
    covered by features) of the video, fps and rfps. Example of a
    annotation file:

    .. code-block:: JSON

        {
            "v_--1DO2V4K74":  {
                "duration_second": 211.53,
                "duration_frame": 6337,
                "annotations": [
                    {
                        "segment": [
                            30.025882995319815,
                            205.2318595943838
                        ],
                        "label": "Rock climbing"
                    }
                ],
                "feature_frame": 6336,
                "fps": 30.0,
                "rfps": 29.9579255898
            },
            "v_--6bJUbfpnQ": {
                "duration_second": 26.75,
                "duration_frame": 647,
                "annotations": [
                    {
                        "segment": [
                            2.578755070202808,
                            24.914101404056165
                        ],
                        "label": "Drinking beer"
                    }
                ],
                "feature_frame": 624,
                "fps": 24.0,
                "rfps": 24.1869158879
            },
            ...
        }


    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        data_prefix (str | None): Path to a directory where videos are held.
            Default: None.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
    """

    def __init__(self, ann_file, pipeline, data_prefix=None, test_mode=False):
        super().__init__(ann_file, pipeline, data_prefix, test_mode)

    def load_annotations(self):
        """Load the annotation according to ann_file into video_infos."""
        video_infos = []
        anno_database = mmcv.load(self.ann_file)
        for video_name in anno_database:
            video_info = anno_database[video_name]
            video_info['video_name'] = video_name
            video_infos.append(video_info)
        return video_infos

    def prepare_test_frames(self, idx):
        """Prepare the frames for testing given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        results['data_prefix'] = self.data_prefix
        return self.pipeline(results)

    def prepare_train_frames(self, idx):
        """Prepare the frames for training given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        results['data_prefix'] = self.data_prefix
        return self.pipeline(results)

    def __len__(self):
        """Get the size of the dataset."""
        return len(self.video_infos)

    def _import_ground_truth(self, activity_index=None):
        """Read ground truth data from video_infos."""
        ground_truth = {}
        for video_info in self.video_infos:
            video_id = video_info['video_name'][2:]
            this_video_ground_truths = []
            for ann in video_info['annotations']:
                t_start, t_end = ann['segment']
                if activity_index:
                    label = activity_index[ann['label']]
                else:
                    label = ann['label']
                this_video_ground_truths.append([t_start, t_end, label])
            ground_truth[video_id] = np.array(this_video_ground_truths)
        return ground_truth

    def proposals2json(self, results, show_progress=False):
        """Convert all proposals to a final dict(json) format.

        Args:
            results (list[dict]): All proposals.
            show_progress (bool): Whether to show the progress bar.
                Defaults: False.

        Returns:
            dict: The final result dict. E.g.

            .. code-block:: Python

                dict(video-1=[dict(segment=[1.1,2.0]. score=0.9),
                              dict(segment=[50.1, 129.3], score=0.6)])
        """
        result_dict = {}
        print('Convert proposals to json format')
        if show_progress:
            prog_bar = mmcv.ProgressBar(len(results))
        for result in results:
            video_name = result['video_name']
            result_dict[video_name[2:]] = result['proposal_list']
            if show_progress:
                prog_bar.update()
        return result_dict

    def _import_proposals(self, results):
        """Read predictions from results."""
        proposals = {}
        num_proposals = 0
        for result in results:
            video_id = result['video_name'][2:]
            this_video_proposals = []
            for proposal in result['proposal_list']:
                t_start, t_end = proposal['segment']
                score = proposal['score']
                this_video_proposals.append([t_start, t_end, score])
                num_proposals += 1
            proposals[video_id] = np.array(this_video_proposals)
        return proposals, num_proposals

    def dump_results(self,
                     results,
                     out,
                     output_format,
                     classifier_input=True,
                     version='VERSION 1.3'):
        """Dump data to json/csv files."""
        if output_format == 'json':
            result_dict = self.proposals2json(results)
            output_dict = {
                'version': version,
                'results': result_dict,
                'external_data': {}
            }
            mmcv.dump(output_dict, out)

            if classifier_input:
                with open(out, 'r') as f_out:
                    proposal_results = json.load(f_out)['results']
                videos = proposal_results.keys()
                video_idx = 0

                out_classifier_input = open(
                    self.ann_file.replace('anet_anno_val.json',
                                          'anet_val_classifier_input.txt'),
                    'w')
                # The activity index file is constructed according to
                # 'https://github.com/activitynet/ActivityNet/blob/master/Evaluation/eval_classification.py'
                activity_index_file = open(
                    self.ann_file.replace('anet_anno_val.json',
                                          'anet_activity_indexes_val.txt'),
                    'r')
                activity_index, class_idx = {}, 0
                for line in activity_index_file.readlines():
                    activity_index[line.strip()] = class_idx
                    class_idx += 1

                ground_truth = self._import_ground_truth(activity_index)
                proposal, num_proposals = self._import_proposals(results)
                with open(self.ann_file.replace('val', 'full'), 'r') as f_full:
                    full_ground_truth = json.load(f_full)

                for video in videos:
                    num_frames = full_ground_truth['v_' +
                                                   video]['duration_frame']
                    fps = full_ground_truth['v_' + video]['fps']
                    tiou, t_overlap = pairwise_temporal_iou(
                        proposal[video][:, :2].astype(float),
                        ground_truth[video][:, :2].astype(float),
                        overlap_self=True)

                    self.dump_classifier_input(video_idx, video, num_frames,
                                               fps, ground_truth[video],
                                               proposal[video], tiou,
                                               t_overlap, out_classifier_input)
                    video_idx += 1

        elif output_format == 'csv':
            # TODO: add csv handler to mmcv and use mmcv.dump
            os.makedirs(out, exist_ok=True)
            header = 'action,start,end,tmin,tmax'
            for result in results:
                video_name, outputs = result
                output_path = osp.join(out, video_name + '.csv')
                np.savetxt(
                    output_path,
                    outputs,
                    header=header,
                    delimiter=',',
                    comments='')
        else:
            raise ValueError(
                f'The output format {output_format} is not supported.')

    def dump_classifier_input(self, video_idx, video, num_frames, fps, gts,
                              proposals, tiou, t_overlap,
                              out_classifier_input):
        out_classifier_input.write('#{}\n{}\n{}\n{}\n{}\n'.format(
            video_idx, video, num_frames, fps, gts.shape[0]))
        for gt in gts:
            out_classifier_input.write('{} {} {}\n'.format(
                int(gt[2]), gt[0], gt[1]))
        out_classifier_input.write('{}\n'.format(proposals.shape[0]))

        best_iou = np.amax(tiou, axis=0)
        best_iou_index = np.argmax(tiou, axis=0)
        best_overlap = np.amax(t_overlap, axis=0)
        best_overlap_index = np.argmax(t_overlap, axis=0)

        for i in range(proposals.shape[0]):
            index_iou = best_iou_index[i]
            index_overlap = best_overlap_index[i]
            label_iou = gts[index_iou][2]
            label_overlap = gts[index_overlap][2]
            if label_iou != label_overlap:
                label = label_iou if label_iou != 0 else label_overlap
            else:
                label = label_iou
            if best_iou[i] == 0 and best_overlap[i] == 0:
                out_classifier_input.write('0 0 0 {} {}\n'.format(
                    proposals[i][0], proposals[i][1]))
            else:
                out_classifier_input.write('{} {} {} {} {}\n'.format(
                    int(label), best_iou[i], best_overlap[i], proposals[i][0],
                    proposals[i][1]))

    def evaluate(
            self,
            results,
            metrics='AR@AN',
            metric_options={
                'AR@AN':
                dict(
                    max_avg_proposals=100,
                    temporal_iou_thresholds=np.linspace(0.5, 0.95, 10))
            },
            logger=None,
            **deprecated_kwargs):
        """Evaluation in feature dataset.

        Args:
            results (list[dict]): Output results.
            metrics (str | sequence[str]): Metrics to be performed.
                Defaults: 'AR@AN'.
            metric_options (dict): Dict for metric options. Options are
                ``max_avg_proposals``, ``temporal_iou_thresholds`` for
                ``AR@AN``.
                default: ``{'AR@AN': dict(max_avg_proposals=100,
                temporal_iou_thresholds=np.linspace(0.5, 0.95, 10))}``.
            logger (logging.Logger | None): Training logger. Defaults: None.
            deprecated_kwargs (dict): Used for containing deprecated arguments.
                See 'https://github.com/open-mmlab/mmaction2/pull/286'.

        Returns:
            dict: Evaluation results for evaluation metrics.
        """
        # Protect ``metric_options`` since it uses mutable value as default
        metric_options = copy.deepcopy(metric_options)

        if deprecated_kwargs != {}:
            warnings.warn(
                'Option arguments for metrics has been changed to '
                "`metric_options`, See 'https://github.com/open-mmlab/mmaction2/pull/286' "  # noqa: E501
                'for more details')
            metric_options['AR@AN'] = dict(metric_options['AR@AN'],
                                           **deprecated_kwargs)

        if not isinstance(results, list):
            raise TypeError(f'results must be a list, but got {type(results)}')
        assert len(results) == len(self), (
            f'The length of results is not equal to the dataset len: '
            f'{len(results)} != {len(self)}')

        metrics = metrics if isinstance(metrics, (list, tuple)) else [metrics]
        allowed_metrics = ['AR@AN']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        eval_results = {}
        ground_truth = self._import_ground_truth()
        proposal, num_proposals = self._import_proposals(results)

        for metric in metrics:
            if metric == 'AR@AN':
                temporal_iou_thresholds = metric_options.setdefault(
                    'AR@AN', {}).setdefault('temporal_iou_thresholds',
                                            np.linspace(0.5, 0.95, 10))
                max_avg_proposals = metric_options.setdefault(
                    'AR@AN', {}).setdefault('max_avg_proposals', 100)
                if isinstance(temporal_iou_thresholds, list):
                    temporal_iou_thresholds = np.array(temporal_iou_thresholds)

                recall, _, _, auc = (
                    average_recall_at_avg_proposals(
                        ground_truth,
                        proposal,
                        num_proposals,
                        max_avg_proposals=max_avg_proposals,
                        temporal_iou_thresholds=temporal_iou_thresholds))
                eval_results['auc'] = auc
                eval_results['AR@1'] = np.mean(recall[:, 0])
                eval_results['AR@5'] = np.mean(recall[:, 4])
                eval_results['AR@10'] = np.mean(recall[:, 9])
                eval_results['AR@100'] = np.mean(recall[:, 99])

        return eval_results
