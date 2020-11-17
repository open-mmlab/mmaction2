from abc import ABCMeta, abstractmethod

import torch

from .sampling_result import SamplingResult


class BaseSampler(metaclass=ABCMeta):
    """Base class for bbox_samplers.

    All datasets to sample bboxes should subclass it.
    All subclasses should overwrite:

    - Methods:`_sample_pos`, sample_positive_bboxes
    - Methods:`_sample_neg`, sample_negative_bboxes

    Args:
        num (int): Number of bboxes to sample.
        pos_fraction (float): Fraction of positive bboxes (expected).
        neg_pos_ub (float): The upper bound of #negative / #positive. If set as
            negative, the constraint will be ignored. Default: -1.
        add_gt_as_proposals (bool): Also add gt bboxes as proposals.
            Default: True.
    """

    def __init__(self,
                 num,
                 pos_fraction,
                 neg_pos_ub=-1,
                 add_gt_as_proposals=True):

        self.num = num
        self.pos_fraction = pos_fraction
        self.neg_pos_ub = neg_pos_ub
        self.add_gt_as_proposals = add_gt_as_proposals

    @abstractmethod
    def _sample_pos(self, assign_result, num_expected):
        pass

    @abstractmethod
    def _sample_neg(self, assign_result, num_expected):
        pass

    def sample(self, assign_result, bboxes, gt_bboxes, gt_labels=None):
        """Sample positive and negative bboxes.
        This is a simple implementation of bbox sampling given candidates,
        assigning results and ground truth bboxes.
        Args:
            assign_result (:obj:`AssignResult`): Bbox assigning results.
            bboxes (Tensor): Boxes to be sampled from.
            gt_bboxes (Tensor): Ground truth bboxes.
            gt_labels (Tensor, optional): Class labels of ground truth bboxes.

        Returns:
            :obj:`SamplingResult`: Sampling result.
        """
        bboxes = bboxes[:, :4]

        gt_flags = bboxes.new_zeros((bboxes.shape[0], ), dtype=torch.uint8)
        if self.add_gt_as_proposals:
            bboxes = torch.cat([gt_bboxes, bboxes], dim=0)
            assign_result.add_gt_(gt_labels)
            gt_ones = bboxes.new_ones(gt_bboxes.shape[0], dtype=torch.uint8)
            gt_flags = torch.cat([gt_ones, gt_flags])

        num_expected_pos = int(self.num * self.pos_fraction)
        pos_inds = self._sample_pos(
            assign_result, num_expected_pos, bboxes=bboxes)

        # We found that sampled indices have duplicated items occasionally.
        # (may be a bug of PyTorch)
        pos_inds = pos_inds.unique()
        num_sampled_pos = pos_inds.numel()
        num_expected_neg = self.num - num_sampled_pos
        if self.neg_pos_ub >= 0:
            _pos = max(1, num_sampled_pos)
            neg_upper_bound = int(self.neg_pos_ub * _pos)
            if num_expected_neg > neg_upper_bound:
                num_expected_neg = neg_upper_bound
        neg_inds = self._sample_neg(
            assign_result, num_expected_neg, bboxes=bboxes)

        neg_inds = neg_inds.unique()

        return SamplingResult(pos_inds, neg_inds, bboxes, gt_bboxes,
                              assign_result, gt_flags)
