import copy
import math
import os.path as osp

import mmcv
import numpy as np
from terminaltables import AsciiTable
from torch.nn.modules.utils import _pair

from ..localization import (eval_ap_parallel, load_localize_proposal_file,
                            perform_regression, process_norm_proposal_file,
                            results_to_detections, temporal_iou, temporal_nms)
from ..localization.ssn_utils import parse_frame_folder
from ..utils import get_root_logger
from .base import BaseDataset
from .registry import DATASETS


class SSNInstance:
    """Proposal instance of SSN.

    Args:
        start_frame (int): Index of the proposal's start frame.
        end_frame (int): Index of the proposal's end frame.
        num_video_frames (int): Total frames of the video.
        label (int): The category label of the proposal. Default: None.
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
        self.end_frame = min(end_frame, num_video_frames + 1)
        self.num_video_frames = num_video_frames
        self.label = label if label is not None else -1
        self.coverage = (end_frame - start_frame) / num_video_frames
        self.best_iou = best_iou
        self.overlap_self = overlap_self
        self.loc_reg = None
        self.size_reg = None
        self.regression_targets = [0., 0.]

    def compute_regression_targets(self, gt_list, positive_threshold):
        """Compute regression targets of positive proposals.

        Args:
            gt_list (list): The list of groundtruth instances.
            positive_threshold (float): Minimum threshold of overlap of
                positive/foreground proposals and groundtruths.
        """
        # Background proposals do not have regression targets.
        if self.best_iou < positive_threshold:
            return

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
        if gt_size / proposal_size > 0:
            self.size_reg = math.log(gt_size / proposal_size)
        else:
            print(gt_size, proposal_size, self.start_frame, self.end_frame)
            raise ValueError('gt_size / proposal_size should be valid.')
        self.regression_targets = ([self.loc_reg, self.size_reg]
                                   if self.loc_reg is not None else [0., 0.])


@DATASETS.register_module()
class SSNDataset(BaseDataset):
    """Proposal frame dataset for Structured Segment Networks. Based on
    proposal information, the dataset loads raw frames and apply specified
    transforms to return a dict containing the frame tensors and other
    information. The ann_file is a text file with multiple lines and each
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
        data_prefix (str): Path to a directory where videos are held.
            Default: None.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
        filename_tmpl (str): Template for each filename.
            Default: 'img_{:05}.jpg'.
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
            of augmentation to that of the proposal. Defualt: (0.5, 0.5).
        clip_len (int): Frames of each sampled output clip.
            Default: 1.
        frame_interval (int): Temporal interval of adjacent sampled frames.
            Default: 1.
        filter_gt (bool): Whether to filter videos with no annotation
            during training. Default: True.
        no_regression (bool): Whether to perform regression.
        verbose (bool): Whether to print full information or not.
            Default: False.
    """

    def __init__(self,
                 ann_file,
                 pipeline,
                 train_cfg,
                 test_cfg,
                 data_prefix=None,
                 test_mode=False,
                 filename_tmpl='img_{:05d}.jpg',
                 video_centric=True,
                 reg_normalize_constants=None,
                 body_segments=5,
                 aug_segments=(2, 2),
                 aug_ratio=(0.5, 0.5),
                 clip_len=1,
                 frame_interval=1,
                 filter_gt=True,
                 no_regression=False,
                 verbose=False):
        self.logger = get_root_logger()
        super(SSNDataset, self).__init__(ann_file, pipeline, data_prefix,
                                         test_mode)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.assigner = train_cfg.ssn.assigner
        self.sampler = train_cfg.ssn.sampler
        self.verbose = verbose
        self.filename_tmpl = filename_tmpl

        if filter_gt or not test_mode:
            valid_inds = self._filter_records()
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
        self.no_regression = no_regression
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
                self.logger.info(f'{self.proposal_file} does not exist.'
                                 f'Converting from {self.ann_file}')
                frame_dict = parse_frame_folder(
                    self.data_prefix, key_func=lambda x: osp.basename(x))
                process_norm_proposal_file(self.ann_file, self.proposal_file,
                                           frame_dict)
                self.logger.info('Finished conversion.')
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
                if int(x[2]) > int(x[1]):
                    ssn_instance = SSNInstance(
                        int(x[1]),
                        int(x[2]),
                        num_frames,
                        label=int(x[0]),
                        best_iou=1.0)
                    gts.append(ssn_instance)
            gts = list(filter(lambda x: x.start_frame < num_frames, gts))
            # proposals:start, end, num_frames, class_label
            # tIoU=best_iou, overlap_self
            proposals = []
            for x in proposal_info[3]:
                if int(x[4]) > int(x[3]):
                    ssn_instance = SSNInstance(
                        int(x[3]),
                        int(x[4]),
                        num_frames,
                        label=int(x[0]),
                        best_iou=float(x[1]),
                        overlap_self=float(x[2]))
                    proposals.append(ssn_instance)
            proposals = list(
                filter(lambda x: x.start_frame < num_frames, proposals))
            video_infos.append(
                dict(
                    frame_dir=frame_dir,
                    video_id=proposal_info[0],
                    total_frames=num_frames,
                    gts=gts,
                    proposals=proposals))
        return video_infos

    def evaluate(self,
                 dataset,
                 results,
                 metrics='mAP',
                 eval='thumos14',
                 **kwargs):
        detections = results_to_detections(dataset, results,
                                           **self.test_cfg.ssn.evaluater)

        if not self.no_regression:
            self.logger.info('Performing location regression')
            for class_idx in range(len(detections)):
                detections[class_idx] = {
                    k: perform_regression(v)
                    for k, v in detections[class_idx].items()
                }
            self.logger.info('Regression finished')

        self.logger.info('Performing NMS')
        for class_idx in range(len(detections)):
            detections[class_idx] = {
                k: temporal_nms(v, self.test_cfg.ssn.evaluater.nms)
                for k, v in detections[class_idx].items()
            }
        self.logger.info('NMS finished')

        iou_range = np.arange(0.1, 1.0, .1)

        # get gts
        all_gts = dataset.get_all_gts()
        for class_idx in range(len(detections)):
            if (class_idx not in all_gts.keys()):
                all_gts[class_idx] = dict()

        # get predictions
        plain_detections = {}
        for class_idx in range(len(detections)):
            detection_list = []
            for vid, dets in detections[class_idx].items():
                detection_list.extend([[vid, class_idx] + x[:3]
                                       for x in dets.tolist()])
            plain_detections[class_idx] = detection_list

        ap_values = eval_ap_parallel(plain_detections, all_gts, iou_range)
        map_iou = ap_values.mean(axis=0)
        self.logger.info('Evaluation finished')

        # display
        display_title = f'Temporal detection performance ({eval})'
        display_data = [['IoU thresh'], ['mean AP']]

        for i in range(len(iou_range)):
            display_data[0].append(f'{iou_range[i]:.02f}')
            display_data[1].append(f'{map_iou[i]:.04f}')
        table = AsciiTable(display_data, display_title)
        table.justify_columns[-1] = 'right'
        table.inner_footing_row_border = True
        self.logger.info(table.table)

    def construct_proposal_pools(self):
        """Construct positve proposal pool, incomplete proposal pool and
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
            vid = video_info['video_id']
            for gt in video_info['gts']:
                class_idx = gt.label - 1
                # gt_info: [relative_start, relative_end]
                gt_info = [
                    gt.start_frame / video_info['total_frames'],
                    gt.end_frame / video_info['total_frames']
                ]
                if class_idx not in gts.keys():
                    gts[class_idx] = dict()
                    gts[class_idx][vid] = []
                    gts[class_idx][vid].append(gt_info)
                else:
                    if vid in gts[class_idx].keys():
                        gts[class_idx][vid].append(gt_info)
                    else:
                        gts[class_idx][vid] = []
                        gts[class_idx][vid].append(gt_info)

        return gts

    def _filter_records(self):
        """Filter videos with no groundtruth during training."""
        valid_inds = []
        for i, video_info in enumerate(self.video_infos):
            if len(video_info['gts']) > 0:
                valid_inds.append(i)

        return valid_inds

    def get_positives(self, gts, proposals, positive_threshold, with_gt=True):
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
            proposal.compute_regression_targets(gts, positive_threshold)

        return positives

    def get_negatives(self,
                      proposals,
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
                of background proposals in video duration.
                Default: 0.01.
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
                list[(video_id, :obj:`SSNInstance`), proposal_type]
            """

            if len(video_pool) == 0:
                idx = np.random.choice(
                    len(dataset_pool), num_requested_proposals, replace=False)
                return [(dataset_pool[x], proposal_type) for x in idx]
            else:
                replicate = len(video_pool) < num_requested_proposals
                idx = np.random.choice(
                    len(video_pool),
                    num_requested_proposals,
                    replace=replicate)
                return [((video_id, video_pool[x]), proposal_type)
                        for x in idx]

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

        out_proposals.extend([(x, 0) for x in np.random.choice(
            self.positive_pool, self.positive_per_video, replace=False)])
        out_proposals.extend([(x, 1) for x in np.random.choice(
            self.incomplete_pool, self.incomplete_per_video, replace=False)])
        out_proposals.extend([(x, 2) for x in np.random.choice(
            self.background_pool, self.background_per_video, replace=False)])

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

        # yapf: disable
        stage_split = [self.aug_segments[0], self.aug_segments[0] + self.body_segments,  # noqa: E501
                       self.aug_segments[0] + self.body_segments + self.aug_segments[1]]  # noqa: E501
        # yapf: enable
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
        results['modality'] = 'RGB'

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

        for idx, proposal in enumerate(results['out_proposals']):
            # proposal: [(video_id, SSNInstance), proposal_type]
            num_frames = proposal[0][1].num_video_frames

            (starting_scale_factor, ending_scale_factor,
             stage_split) = self._get_stage(proposal[0][1], num_frames)

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
            if not isinstance(proposal[1], int):
                raise TypeError(f'proposal_type must be an int,'
                                f'but got {type(proposal[1])}')
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
        results['modality'] = 'RGB'

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

            proposal_ticks = (int(real_relative_starting * num_sampled_frames),
                              int(relative_proposal[0] * num_sampled_frames),
                              int(relative_proposal[1] * num_sampled_frames),
                              int(real_relative_ending * num_sampled_frames))

            relative_proposal_list.append(relative_proposal)
            proposal_tick_list.append(proposal_ticks)
            scale_factor_list.append(
                (starting_scale_factor, ending_scale_factor))

        results['relative_proposal_list'] = np.array(
            relative_proposal_list, dtype=np.float32)
        results['scale_factor_list'] = np.array(
            scale_factor_list, dtype=np.float32)
        results['proposal_tick_list'] = np.array(proposal_tick_list)
        results['reg_norm_consts'] = self.reg_norm_consts

        return self.pipeline(results)
