import math

import numpy as np

from mmaction.localization.proposal_utils import temporal_iou


class SSNInstance(object):

    def __init__(self,
                 start_frame,
                 end_frame,
                 num_frames,
                 label=None,
                 best_iou=None,
                 overlap_self=None):
        self.start_frame = start_frame
        self.end_frame = min(end_frame, num_frames)
        self.label = label if label is not None else -1
        self.coverage = (end_frame - start_frame) / num_frames
        self.best_iou = best_iou
        self.overlap_self = overlap_self
        self.loc_reg = None
        self.size_reg = None
        self.regression_targets = ([self.loc_reg, self.size_reg]
                                   if self.loc_reg is not None else [0., 0.])

    def compute_regression_targets(self, gt_list, fg_thresh):
        # background proposals do not need this
        if self.best_iou < fg_thresh:
            return

        # find the groundtruth instance with the highest IOU
        ious = []
        for gt in gt_list:
            ious.append(
                temporal_iou(self.start_frame, self.end_frame, gt.start_frame,
                             gt.end_frame))
        best_gt = gt_list[np.argmax(ious)]

        prop_center = (self.start_frame + self.end_frame) / 2
        gt_center = (best_gt.start_frame + best_gt.end_frame) / 2
        prop_size = self.end_frame - self.start_frame + 1
        gt_size = best_gt.end_frame - best_gt.start_frame + 1

        # get regression target:
        # (1). center shift proportional to the proposal duration
        # (2). logarithm of the groundtruth duration over proposal duration
        self.loc_reg = (gt_center - prop_center) / prop_size
        try:
            self.size_reg = math.log(gt_size / prop_size)
        except ValueError:
            print(gt_size, prop_size, self.start_frame, self.end_frame)
            raise ValueError('gt_size / prop_size should be valid.')
        self.regression_targets = ([self.loc_reg, self.size_reg]
                                   if self.loc_reg is not None else [0., 0.])
