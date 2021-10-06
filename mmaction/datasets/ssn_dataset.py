# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
import warnings
from collections import OrderedDict

import mmcv
import numpy as np
from torch.nn.modules.utils import _pair

from ..core import softmax
from ..localization import (eval_ap, load_localize_proposal_file,
                            perform_regression, temporal_iou, temporal_nms)
from ..utils import get_root_logger
from .base import BaseDataset
from .builder import DATASETS


class SSNInstance:
    """Proposal instance of SSN.

    Args:
        start_frame (int): Index of the proposal's start frame.
        end_frame (int): Index of the proposal's end frame.
        num_video_frames (int): Total frames of the video.
        label (int | None): The category label of the proposal. Default: None.
        best_iou (float): The highest IOU with the groundtruth instance.
            Default: 0.
        overlap_self (float): Percent of the proposal's own span contained
            in a groundtruth instance. Default: 0.
    """

    def __init__(self,
                 start_frame,
                 end_frame,
                 num_video_frames,
                 label=None,
                 best_iou=0,
                 overlap_self=0):
        self.start_frame = start_frame
        self.end_frame = min(end_frame, num_video_frames)
        self.num_video_frames = num_video_frames
        self.label = label if label is not None else -1
        self.coverage = (end_frame - start_frame) / num_video_frames
        self.best_iou = best_iou
        self.overlap_self = overlap_self
        self.loc_reg = None
        self.size_reg = None
        self.regression_targets = [0., 0.]

    def compute_regression_targets(self, gt_list):
        """Compute regression targets of positive proposals.

        Args:
            gt_list (list): The list of groundtruth instances.
        """
        # Find the groundtruth instance with the highest IOU.
        ious = [
            temporal_iou(self.start_frame, self.end_frame, gt.start_frame,
                         gt.end_frame) for gt in gt_list
        ]
        best_gt = gt_list[np.argmax(ious)]

        # interval: [start_frame, end_frame)
        proposal_center = (self.start_frame + self.end_frame - 1) / 2
        gt_center = (best_gt.start_frame + best_gt.end_frame - 1) / 2
        proposal_size = self.end_frame - self.start_frame
        gt_size = best_gt.end_frame - best_gt.start_frame

        # Get regression targets:
        # (1). Localization regression target:
        #     center shift proportional to the proposal duration
        # (2). Duration/Size regression target:
        #     logarithm of the groundtruth duration over proposal duration

        self.loc_reg = (gt_center - proposal_center) / proposal_size
        self.size_reg = np.log(gt_size / proposal_size)
        self.regression_targets = ([self.loc_reg, self.size_reg]
                                   if self.loc_reg is not None else [0., 0.])


@DATASETS.register_module()
class SSNDataset(BaseDataset):
    """Proposal frame dataset for Structured Segment Networks.

    Based on proposal information, the dataset loads raw frames and applies
    specified transforms to return a dict containing the frame tensors and
    other information.

    The ann_file is a text file with multiple lines and each
    video's information takes up several lines. This file can be a normalized
    file with percent or standard file with specific frame indexes. If the file
    is a normalized file, it will be converted into a standard file first.

    Template information of a video in a standard file:
    .. code-block:: txt
        # index
        video_id
        num_frames
        fps
        num_gts
        label, start_frame, end_frame
        label, start_frame, end_frame
        ...
        num_proposals
        label, best_iou, overlap_self, start_frame, end_frame
        label, best_iou, overlap_self, start_frame, end_frame
        ...

    Example of a standard annotation file:
    .. code-block:: txt
        # 0
        video_validation_0000202
        5666
        1
        3
        8 130 185
        8 832 1136
        8 1303 1381
        5
        8 0.0620 0.0620 790 5671
        8 0.1656 0.1656 790 2619
        8 0.0833 0.0833 3945 5671
        8 0.0960 0.0960 4173 5671
        8 0.0614 0.0614 3327 5671

    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        train_cfg (dict): Config for training.
        test_cfg (dict): Config for testing.
        data_prefix (str): Path to a directory where videos are held.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
        filename_tmpl (str): Template for each filename.
            Default: 'img_{:05}.jpg'.
        start_index (int): Specify a start index for frames in consideration of
            different filename format. Default: 1.
        modality (str): Modality of data. Support 'RGB', 'Flow'.
            Default: 'RGB'.
        video_centric (bool): Whether to sample proposals just from
            this video or sample proposals randomly from the entire dataset.
            Default: True.
        reg_normalize_constants (list): Regression target normalized constants,
            including mean and standard deviation of location and duration.
        body_segments (int): Number of segments in course period.
            Default: 5.
        aug_segments (list[int]): Number of segments in starting and
            ending period. Default: (2, 2).
        aug_ratio (int | float | tuple[int | float]): The ratio of the length
            of augmentation to that of the proposal. Default: (0.5, 0.5).
        clip_len (int): Frames of each sampled output clip.
            Default: 1.
        frame_interval (int): Temporal interval of adjacent sampled frames.
            Default: 1.
        filter_gt (bool): Whether to filter videos with no annotation
            during training. Default: True.
        use_regression (bool): Whether to perform regression. Default: True.
        verbose (bool): Whether to print full information or not.
            Default: False.
    """

    def __init__(self,
                 ann_file,
                 pipeline,
                 train_cfg,
                 test_cfg,
                 data_prefix,
                 test_mode=False,
                 filename_tmpl='img_{:05d}.jpg',
                 start_index=1,
                 modality='RGB',
                 video_centric=True,
                 reg_normalize_constants=None,
                 body_segments=5,
                 aug_segments=(2, 2),
                 aug_ratio=(0.5, 0.5),
                 clip_len=1,
                 frame_interval=1,
                 filter_gt=True,
                 use_regression=True,
                 verbose=False):
        self.logger = get_root_logger()
        super().__init__(
            ann_file,
            pipeline,
            data_prefix=data_prefix,
            test_mode=test_mode,
            start_index=start_index,
            modality=modality)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.assigner = train_cfg.ssn.assigner
        self.sampler = train_cfg.ssn.sampler
        self.evaluater = test_cfg.ssn.evaluater
        self.verbose = verbose
        self.filename_tmpl = filename_tmpl

        if filter_gt or not test_mode:
            valid_inds = [
                i for i, video_info in enumerate(self.video_infos)
                if len(video_info['gts']) > 0
            ]
        self.logger.info(f'{len(valid_inds)} out of {len(self.video_infos)} '
                         f'videos are valid.')
        self.video_infos = [self.video_infos[i] for i in valid_inds]

        # construct three pools:
        # 1. Positive(Foreground)
        # 2. Background
        # 3. Incomplete
        self.positive_pool = []
        self.background_pool = []
        self.incomplete_pool = []
        self.construct_proposal_pools()

        if reg_normalize_constants is None:
            self.reg_norm_consts = self._compute_reg_normalize_constants()
        else:
            self.reg_norm_consts = reg_normalize_constants
        self.video_centric = video_centric
        self.body_segments = body_segments
        self.aug_segments = aug_segments
        self.aug_ratio = _pair(aug_ratio)
        if not mmcv.is_tuple_of(self.aug_ratio, (int, float)):
            raise TypeError(f'aug_ratio should be int, float'
                            f'or tuple of int and float, '
                            f'but got {type(aug_ratio)}')
        assert len(self.aug_ratio) == 2

        total_ratio = (
            self.sampler.positive_ratio + self.sampler.background_ratio +
            self.sampler.incomplete_ratio)
        self.positive_per_video = int(
            self.sampler.num_per_video *
            (self.sampler.positive_ratio / total_ratio))
        self.background_per_video = int(
            self.sampler.num_per_video *
            (self.sampler.background_ratio / total_ratio))
        self.incomplete_per_video = (
            self.sampler.num_per_video - self.positive_per_video -
            self.background_per_video)

        self.test_interval = self.test_cfg.ssn.sampler.test_interval
        # number of consecutive frames
        self.clip_len = clip_len
        # number of steps (sparse sampling for efficiency of io)
        self.frame_interval = frame_interval

        # test mode or not
        self.filter_gt = filter_gt
        self.use_regression = use_regression
        self.test_mode = test_mode

        # yapf: disable
        if self.verbose:
            self.logger.info(f"""
            SSNDataset: proposal file {self.proposal_file} parsed.

            There are {len(self.positive_pool) + len(self.background_pool) +
                len(self.incomplete_pool)} usable proposals from {len(self.video_infos)} videos.
            {len(self.positive_pool)} positive proposals
            {len(self.incomplete_pool)} incomplete proposals
            {len(self.background_pool)} background proposals

            Sample config:
            FG/BG/INCOMP: {self.positive_per_video}/{self.background_per_video}/{self.incomplete_per_video}  # noqa:E501
            Video Centric: {self.video_centric}

            Regression Normalization Constants:
            Location: mean {self.reg_norm_consts[0][0]:.05f} std {self.reg_norm_consts[1][0]:.05f} # noqa: E501
            Duration: mean {self.reg_norm_consts[0][1]:.05f} std {self.reg_norm_consts[1][1]:.05f} # noqa: E501
            """)
        # yapf: enable
        else:
            self.logger.info(
                f'SSNDataset: proposal file {self.proposal_file} parsed.')

    def load_annotations(self):
        """Load annotation file to get video information."""
        video_infos = []
        if 'normalized_' in self.ann_file:
            self.proposal_file = self.ann_file.replace('normalized_', '')
            if not osp.exists(self.proposal_file):
                raise Exception(f'Please refer to `$MMACTION2/tools/data` to'
                                f'denormalize {self.ann_file}.')
        else:
            self.proposal_file = self.ann_file
        proposal_infos = load_localize_proposal_file(self.proposal_file)
        # proposal_info:[video_id, num_frames, gt_list, proposal_list]
        # gt_list member: [label, start_frame, end_frame]
        # proposal_list member: [label, best_iou, overlap_self,
        #                        start_frame, end_frame]
        for proposal_info in proposal_infos:
            if self.data_prefix is not None:
                frame_dir = osp.join(self.data_prefix, proposal_info[0])
            num_frames = int(proposal_info[1])
            # gts:start, end, num_frames, class_label, tIoU=1
            gts = []
            for x in proposal_info[2]:
                if int(x[2]) > int(x[1]) and int(x[1]) < num_frames:
                    ssn_instance = SSNInstance(
                        int(x[1]),
                        int(x[2]),
                        num_frames,
                        label=int(x[0]),
                        best_iou=1.0)
                    gts.append(ssn_instance)
            # proposals:start, end, num_frames, class_label
            # tIoU=best_iou, overlap_self
            proposals = []
            for x in proposal_info[3]:
                if int(x[4]) > int(x[3]) and int(x[3]) < num_frames:
                    ssn_instance = SSNInstance(
                        int(x[3]),
                        int(x[4]),
                        num_frames,
                        label=int(x[0]),
                        best_iou=float(x[1]),
                        overlap_self=float(x[2]))
                    proposals.append(ssn_instance)
            video_infos.append(
                dict(
                    frame_dir=frame_dir,
                    video_id=proposal_info[0],
                    total_frames=num_frames,
                    gts=gts,
                    proposals=proposals))
        return video_infos

    def results_to_detections(self, results, top_k=2000, **kwargs):
        """Convert prediction results into detections.

        Args:
            results (list): Prediction results.
            top_k (int): Number of top results. Default: 2000.

        Returns:
            list: Detection results.
        """
        num_classes = results[0]['activity_scores'].shape[1] - 1
        detections = [dict() for _ in range(num_classes)]

        for idx in range(len(self)):
            video_id = self.video_infos[idx]['video_id']
            relative_proposals = results[idx]['relative_proposal_list']
            if len(relative_proposals[0].shape) == 3:
                relative_proposals = np.squeeze(relative_proposals, 0)

            activity_scores = results[idx]['activity_scores']
            completeness_scores = results[idx]['completeness_scores']
            regression_scores = results[idx]['bbox_preds']
            if regression_scores is None:
                regression_scores = np.zeros(
                    (len(relative_proposals), num_classes, 2),
                    dtype=np.float32)
            regression_scores = regression_scores.reshape((-1, num_classes, 2))

            if top_k <= 0:
                combined_scores = (
                    softmax(activity_scores[:, 1:], dim=1) *
                    np.exp(completeness_scores))
                for i in range(num_classes):
                    center_scores = regression_scores[:, i, 0][:, None]
                    duration_scores = regression_scores[:, i, 1][:, None]
                    detections[i][video_id] = np.concatenate(
                        (relative_proposals, combined_scores[:, i][:, None],
                         center_scores, duration_scores),
                        axis=1)
            else:
                combined_scores = (
                    softmax(activity_scores[:, 1:], dim=1) *
                    np.exp(completeness_scores))
                keep_idx = np.argsort(combined_scores.ravel())[-top_k:]
                for k in keep_idx:
                    class_idx = k % num_classes
                    proposal_idx = k // num_classes
                    new_item = [
                        relative_proposals[proposal_idx, 0],
                        relative_proposals[proposal_idx,
                                           1], combined_scores[proposal_idx,
                                                               class_idx],
                        regression_scores[proposal_idx, class_idx,
                                          0], regression_scores[proposal_idx,
                                                                class_idx, 1]
                    ]
                    if video_id not in detections[class_idx]:
                        detections[class_idx][video_id] = np.array([new_item])
                    else:
                        detections[class_idx][video_id] = np.vstack(
                            [detections[class_idx][video_id], new_item])

        return detections

    def evaluate(self,
                 results,
                 metrics='mAP',
                 metric_options=dict(mAP=dict(eval_dataset='thumos14')),
                 logger=None,
                 **deprecated_kwargs):
        """Evaluation in SSN proposal dataset.

        Args:
            results (list[dict]): Output results.
            metrics (str | sequence[str]): Metrics to be performed.
                Defaults: 'mAP'.
            metric_options (dict): Dict for metric options. Options are
                ``eval_dataset`` for ``mAP``.
                Default: ``dict(mAP=dict(eval_dataset='thumos14'))``.
            logger (logging.Logger | None): Logger for recording.
                Default: None.
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
            metric_options['mAP'] = dict(metric_options['mAP'],
                                         **deprecated_kwargs)

        if not isinstance(results, list):
            raise TypeError(f'results must be a list, but got {type(results)}')
        assert len(results) == len(self), (
            f'The length of results is not equal to the dataset len: '
            f'{len(results)} != {len(self)}')

        metrics = metrics if isinstance(metrics, (list, tuple)) else [metrics]
        allowed_metrics = ['mAP']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        detections = self.results_to_detections(results, **self.evaluater)

        if self.use_regression:
            self.logger.info('Performing location regression')
            for class_idx, _ in enumerate(detections):
                detections[class_idx] = {
                    k: perform_regression(v)
                    for k, v in detections[class_idx].items()
                }
            self.logger.info('Regression finished')

        self.logger.info('Performing NMS')
        for class_idx, _ in enumerate(detections):
            detections[class_idx] = {
                k: temporal_nms(v, self.evaluater.nms)
                for k, v in detections[class_idx].items()
            }
        self.logger.info('NMS finished')

        # get gts
        all_gts = self.get_all_gts()
        for class_idx, _ in enumerate(detections):
            if class_idx not in all_gts:
                all_gts[class_idx] = dict()

        # get predictions
        plain_detections = {}
        for class_idx, _ in enumerate(detections):
            detection_list = []
            for video, dets in detections[class_idx].items():
                detection_list.extend([[video, class_idx] + x[:3]
                                       for x in dets.tolist()])
            plain_detections[class_idx] = detection_list

        eval_results = OrderedDict()
        for metric in metrics:
            if metric == 'mAP':
                eval_dataset = metric_options.setdefault('mAP', {}).setdefault(
                    'eval_dataset', 'thumos14')
                if eval_dataset == 'thumos14':
                    iou_range = np.arange(0.1, 1.0, .1)
                    ap_values = eval_ap(plain_detections, all_gts, iou_range)
                    map_ious = ap_values.mean(axis=0)
                    self.logger.info('Evaluation finished')

                    for iou, map_iou in zip(iou_range, map_ious):
                        eval_results[f'mAP@{iou:.02f}'] = map_iou

        return eval_results

    def construct_proposal_pools(self):
        """Construct positive proposal pool, incomplete proposal pool and
        background proposal pool of the entire dataset."""
        for video_info in self.video_infos:
            positives = self.get_positives(
                video_info['gts'], video_info['proposals'],
                self.assigner.positive_iou_threshold,
                self.sampler.add_gt_as_proposals)
            self.positive_pool.extend([(video_info['video_id'], proposal)
                                       for proposal in positives])

            incompletes, backgrounds = self.get_negatives(
                video_info['proposals'],
                self.assigner.incomplete_iou_threshold,
                self.assigner.background_iou_threshold,
                self.assigner.background_coverage_threshold,
                self.assigner.incomplete_overlap_threshold)
            self.incomplete_pool.extend([(video_info['video_id'], proposal)
                                         for proposal in incompletes])
            self.background_pool.extend([video_info['video_id'], proposal]
                                        for proposal in backgrounds)

    def get_all_gts(self):
        """Fetch groundtruth instances of the entire dataset."""
        gts = {}
        for video_info in self.video_infos:
            video = video_info['video_id']
            for gt in video_info['gts']:
                class_idx = gt.label - 1
                # gt_info: [relative_start, relative_end]
                gt_info = [
                    gt.start_frame / video_info['total_frames'],
                    gt.end_frame / video_info['total_frames']
                ]
                gts.setdefault(class_idx, {}).setdefault(video,
                                                         []).append(gt_info)

        return gts

    @staticmethod
    def get_positives(gts, proposals, positive_threshold, with_gt=True):
        """Get positive/foreground proposals.

        Args:
            gts (list): List of groundtruth instances(:obj:`SSNInstance`).
            proposals (list): List of proposal instances(:obj:`SSNInstance`).
            positive_threshold (float): Minimum threshold of overlap of
                positive/foreground proposals and groundtruths.
            with_gt (bool): Whether to include groundtruth instances in
                positive proposals. Default: True.

        Returns:
            list[:obj:`SSNInstance`]: (positives), positives is a list
                comprised of positive proposal instances.
        """
        positives = [
            proposal for proposal in proposals
            if proposal.best_iou > positive_threshold
        ]

        if with_gt:
            positives.extend(gts)

        for proposal in positives:
            proposal.compute_regression_targets(gts)

        return positives

    @staticmethod
    def get_negatives(proposals,
                      incomplete_iou_threshold,
                      background_iou_threshold,
                      background_coverage_threshold=0.01,
                      incomplete_overlap_threshold=0.7):
        """Get negative proposals, including incomplete proposals and
        background proposals.

        Args:
            proposals (list): List of proposal instances(:obj:`SSNInstance`).
            incomplete_iou_threshold (float): Maximum threshold of overlap
                of incomplete proposals and groundtruths.
            background_iou_threshold (float): Maximum threshold of overlap
                of background proposals and groundtruths.
            background_coverage_threshold (float): Minimum coverage
                of background proposals in video duration. Default: 0.01.
            incomplete_overlap_threshold (float): Minimum percent of incomplete
                proposals' own span contained in a groundtruth instance.
                Default: 0.7.

        Returns:
            list[:obj:`SSNInstance`]: (incompletes, backgrounds), incompletes
                and backgrounds are lists comprised of incomplete
                proposal instances and background proposal instances.
        """
        incompletes = []
        backgrounds = []

        for proposal in proposals:
            if (proposal.best_iou < incomplete_iou_threshold
                    and proposal.overlap_self > incomplete_overlap_threshold):
                incompletes.append(proposal)
            elif (proposal.best_iou < background_iou_threshold
                  and proposal.coverage > background_coverage_threshold):
                backgrounds.append(proposal)

        return incompletes, backgrounds

    def _video_centric_sampling(self, record):
        """Sample proposals from the this video instance.

        Args:
            record (dict): Information of the video instance(video_info[idx]).
                key: frame_dir, video_id, total_frames,
                gts: List of groundtruth instances(:obj:`SSNInstance`).
                proposals: List of proposal instances(:obj:`SSNInstance`).
        """
        positives = self.get_positives(record['gts'], record['proposals'],
                                       self.assigner.positive_iou_threshold,
                                       self.sampler.add_gt_as_proposals)
        incompletes, backgrounds = self.get_negatives(
            record['proposals'], self.assigner.incomplete_iou_threshold,
            self.assigner.background_iou_threshold,
            self.assigner.background_coverage_threshold,
            self.assigner.incomplete_overlap_threshold)

        def sample_video_proposals(proposal_type, video_id, video_pool,
                                   num_requested_proposals, dataset_pool):
            """This method will sample proposals from the this video pool. If
            the video pool is empty, it will fetch from the dataset pool
            (collect proposal of the entire dataset).

            Args:
                proposal_type (int): Type id of proposal.
                    Positive/Foreground: 0
                    Negative:
                        Incomplete: 1
                        Background: 2
                video_id (str): Name of the video.
                video_pool (list): Pool comprised of proposals in this video.
                num_requested_proposals (int): Number of proposals
                    to be sampled.
                dataset_pool (list): Proposals of the entire dataset.

            Returns:
                list[(str, :obj:`SSNInstance`), int]:
                    video_id (str): Name of the video.
                    :obj:`SSNInstance`: Instance of class SSNInstance.
                    proposal_type (int): Type of proposal.
            """

            if len(video_pool) == 0:
                idx = np.random.choice(
                    len(dataset_pool), num_requested_proposals, replace=False)
                return [(dataset_pool[x], proposal_type) for x in idx]

            replicate = len(video_pool) < num_requested_proposals
            idx = np.random.choice(
                len(video_pool), num_requested_proposals, replace=replicate)
            return [((video_id, video_pool[x]), proposal_type) for x in idx]

        out_proposals = []
        out_proposals.extend(
            sample_video_proposals(0, record['video_id'], positives,
                                   self.positive_per_video,
                                   self.positive_pool))
        out_proposals.extend(
            sample_video_proposals(1, record['video_id'], incompletes,
                                   self.incomplete_per_video,
                                   self.incomplete_pool))
        out_proposals.extend(
            sample_video_proposals(2, record['video_id'], backgrounds,
                                   self.background_per_video,
                                   self.background_pool))

        return out_proposals

    def _random_sampling(self):
        """Randomly sample proposals from the entire dataset."""
        out_proposals = []

        positive_idx = np.random.choice(
            len(self.positive_pool),
            self.positive_per_video,
            replace=len(self.positive_pool) < self.positive_per_video)
        out_proposals.extend([(self.positive_pool[x], 0)
                              for x in positive_idx])
        incomplete_idx = np.random.choice(
            len(self.incomplete_pool),
            self.incomplete_per_video,
            replace=len(self.incomplete_pool) < self.incomplete_per_video)
        out_proposals.extend([(self.incomplete_pool[x], 1)
                              for x in incomplete_idx])
        background_idx = np.random.choice(
            len(self.background_pool),
            self.background_per_video,
            replace=len(self.background_pool) < self.background_per_video)
        out_proposals.extend([(self.background_pool[x], 2)
                              for x in background_idx])

        return out_proposals

    def _get_stage(self, proposal, num_frames):
        """Fetch the scale factor of starting and ending stage and get the
        stage split.

        Args:
            proposal (:obj:`SSNInstance`): Proposal instance.
            num_frames (int): Total frames of the video.

        Returns:
            tuple[float, float, list]: (starting_scale_factor,
                ending_scale_factor, stage_split), starting_scale_factor is
                the ratio of the effective sampling length to augment length
                in starting stage, ending_scale_factor is the ratio of the
                effective sampling length to augment length in ending stage,
                stage_split is  ending segment id of starting, course and
                ending stage.
        """
        # proposal interval: [start_frame, end_frame)
        start_frame = proposal.start_frame
        end_frame = proposal.end_frame
        ori_clip_len = self.clip_len * self.frame_interval

        duration = end_frame - start_frame
        assert duration != 0

        valid_starting = max(0,
                             start_frame - int(duration * self.aug_ratio[0]))
        valid_ending = min(num_frames - ori_clip_len + 1,
                           end_frame - 1 + int(duration * self.aug_ratio[1]))

        valid_starting_length = start_frame - valid_starting - ori_clip_len
        valid_ending_length = (valid_ending - end_frame + 1) - ori_clip_len

        starting_scale_factor = ((valid_starting_length + ori_clip_len + 1) /
                                 (duration * self.aug_ratio[0]))
        ending_scale_factor = (valid_ending_length + ori_clip_len + 1) / (
            duration * self.aug_ratio[1])

        aug_start, aug_end = self.aug_segments
        stage_split = [
            aug_start, aug_start + self.body_segments,
            aug_start + self.body_segments + aug_end
        ]

        return starting_scale_factor, ending_scale_factor, stage_split

    def _compute_reg_normalize_constants(self):
        """Compute regression target normalized constants."""
        if self.verbose:
            self.logger.info('Compute regression target normalized constants')
        targets = []
        for video_info in self.video_infos:
            positives = self.get_positives(
                video_info['gts'], video_info['proposals'],
                self.assigner.positive_iou_threshold, False)
            for positive in positives:
                targets.append(list(positive.regression_targets))

        return np.array((np.mean(targets, axis=0), np.std(targets, axis=0)))

    def prepare_train_frames(self, idx):
        """Prepare the frames for training given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        results['filename_tmpl'] = self.filename_tmpl
        results['modality'] = self.modality
        results['start_index'] = self.start_index

        if self.video_centric:
            # yapf: disable
            results['out_proposals'] = self._video_centric_sampling(self.video_infos[idx])  # noqa: E501
            # yapf: enable
        else:
            results['out_proposals'] = self._random_sampling()

        out_proposal_scale_factor = []
        out_proposal_type = []
        out_proposal_labels = []
        out_proposal_reg_targets = []

        for _, proposal in enumerate(results['out_proposals']):
            # proposal: [(video_id, SSNInstance), proposal_type]
            num_frames = proposal[0][1].num_video_frames

            (starting_scale_factor, ending_scale_factor,
             _) = self._get_stage(proposal[0][1], num_frames)

            # proposal[1]: Type id of proposal.
            # Positive/Foreground: 0
            # Negative:
            #   Incomplete: 1
            #   Background: 2

            # Positivte/Foreground proposal
            if proposal[1] == 0:
                label = proposal[0][1].label
            # Incomplete proposal
            elif proposal[1] == 1:
                label = proposal[0][1].label
            # Background proposal
            elif proposal[1] == 2:
                label = 0
            else:
                raise ValueError(f'Proposal type should be 0, 1, or 2,'
                                 f'but got {proposal[1]}')
            out_proposal_scale_factor.append(
                [starting_scale_factor, ending_scale_factor])
            if not isinstance(label, int):
                raise TypeError(f'proposal_label must be an int,'
                                f'but got {type(label)}')
            out_proposal_labels.append(label)
            out_proposal_type.append(proposal[1])

            reg_targets = proposal[0][1].regression_targets
            if proposal[1] == 0:
                # Normalize regression targets of positive proposals.
                reg_targets = ((reg_targets[0] - self.reg_norm_consts[0][0]) /
                               self.reg_norm_consts[1][0],
                               (reg_targets[1] - self.reg_norm_consts[0][1]) /
                               self.reg_norm_consts[1][1])
            out_proposal_reg_targets.append(reg_targets)

        results['reg_targets'] = np.array(
            out_proposal_reg_targets, dtype=np.float32)
        results['proposal_scale_factor'] = np.array(
            out_proposal_scale_factor, dtype=np.float32)
        results['proposal_labels'] = np.array(out_proposal_labels)
        results['proposal_type'] = np.array(out_proposal_type)

        return self.pipeline(results)

    def prepare_test_frames(self, idx):
        """Prepare the frames for testing given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        results['filename_tmpl'] = self.filename_tmpl
        results['modality'] = self.modality
        results['start_index'] = self.start_index

        proposals = results['proposals']
        num_frames = results['total_frames']
        ori_clip_len = self.clip_len * self.frame_interval
        frame_ticks = np.arange(
            0, num_frames - ori_clip_len, self.test_interval, dtype=int) + 1

        num_sampled_frames = len(frame_ticks)

        if len(proposals) == 0:
            proposals.append(SSNInstance(0, num_frames - 1, num_frames))

        relative_proposal_list = []
        proposal_tick_list = []
        scale_factor_list = []

        for proposal in proposals:
            relative_proposal = (proposal.start_frame / num_frames,
                                 proposal.end_frame / num_frames)
            relative_duration = relative_proposal[1] - relative_proposal[0]
            relative_starting_duration = relative_duration * self.aug_ratio[0]
            relative_ending_duration = relative_duration * self.aug_ratio[1]
            relative_starting = (
                relative_proposal[0] - relative_starting_duration)
            relative_ending = relative_proposal[1] + relative_ending_duration

            real_relative_starting = max(0.0, relative_starting)
            real_relative_ending = min(1.0, relative_ending)

            starting_scale_factor = (
                (relative_proposal[0] - real_relative_starting) /
                relative_starting_duration)
            ending_scale_factor = (
                (real_relative_ending - relative_proposal[1]) /
                relative_ending_duration)

            proposal_ranges = (real_relative_starting, *relative_proposal,
                               real_relative_ending)
            proposal_ticks = (np.array(proposal_ranges) *
                              num_sampled_frames).astype(np.int32)

            relative_proposal_list.append(relative_proposal)
            proposal_tick_list.append(proposal_ticks)
            scale_factor_list.append(
                (starting_scale_factor, ending_scale_factor))

        results['relative_proposal_list'] = np.array(
            relative_proposal_list, dtype=np.float32)
        results['scale_factor_list'] = np.array(
            scale_factor_list, dtype=np.float32)
        results['proposal_tick_list'] = np.array(
            proposal_tick_list, dtype=np.int32)
        results['reg_norm_consts'] = self.reg_norm_consts

        return self.pipeline(results)
