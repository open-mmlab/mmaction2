import numpy as np
import torch
import warnings

from ..registry import METRICS
from .base import BaseMetrics
from .overlaps import pairwise_temporal_iou

try:
    from mmdet.core import bbox_overlaps
except (ImportError, ModuleNotFoundError):
    warnings.warn('Please install mmdet to use bbox_overlaps')


def _recalls(all_ious, proposal_nums, thrs):

    img_num = all_ious.shape[0]
    total_gt_num = sum([ious.shape[0] for ious in all_ious])

    ious_ = np.zeros((proposal_nums.size, total_gt_num), dtype=np.float32)
    for k, proposal_num in enumerate(proposal_nums):
        tmp_ious = np.zeros(0)
        for i in range(img_num):
            ious = all_ious[i][:, :proposal_num].copy()
            gt_ious = np.zeros(ious.shape[0])
            if ious.size == 0:
                tmp_ious = np.hstack((tmp_ious, gt_ious))
                continue
            for j in range(ious.shape[0]):
                gt_max_overlaps = ious.argmax(axis=1)
                max_ious = ious[np.arange(0, ious.shape[0]), gt_max_overlaps]
                gt_idx = max_ious.argmax()
                gt_ious[j] = max_ious[gt_idx]
                box_idx = gt_max_overlaps[gt_idx]
                ious[gt_idx, :] = -1
                ious[:, box_idx] = -1
            tmp_ious = np.hstack((tmp_ious, gt_ious))
        ious_[k, :] = tmp_ious

    ious_ = np.fliplr(np.sort(ious_, axis=1))
    recalls = np.zeros((proposal_nums.size, thrs.size))
    for i, thr in enumerate(thrs):
        recalls[:, i] = (ious_ >= thr).sum(axis=1) / float(total_gt_num)

    return recalls


def set_recall_param(proposal_nums, iou_thrs):
    """Check proposal_nums and iou_thrs and set correct format."""
    if isinstance(proposal_nums, list):
        proposal_nums_ = np.array(proposal_nums)
    elif isinstance(proposal_nums, int):
        proposal_nums_ = np.array([proposal_nums])
    else:
        proposal_nums_ = proposal_nums

    if iou_thrs is None:
        _iou_thrs = np.array([0.5])
    elif isinstance(iou_thrs, list):
        _iou_thrs = np.array(iou_thrs)
    elif isinstance(iou_thrs, float):
        _iou_thrs = np.array([iou_thrs])
    else:
        _iou_thrs = iou_thrs

    return proposal_nums_, _iou_thrs


def eval_recalls(gts, proposals, proposal_nums=None, iou_thrs=None):
    """Calculate recalls.

    Args:
        gts(list or ndarray): a list of arrays of shape (n, 4)
        proposals(list or ndarray): a list of arrays of shape (k, 4) or (k, 5)
        proposal_nums(int or list of int or ndarray): top N proposals
        thrs(float or list or ndarray): iou thresholds
    Returns:
        ndarray: recalls of different ious and proposal nums
    """

    img_num = len(gts)
    assert img_num == len(proposals)

    proposal_nums, iou_thrs = set_recall_param(proposal_nums, iou_thrs)

    all_ious = []
    for i in range(img_num):
        if proposals[i].ndim == 2 and proposals[i].shape[1] == 5:
            scores = proposals[i][:, 4]
            sort_idx = np.argsort(scores)[::-1]
            img_proposal = proposals[i][sort_idx, :]
        else:
            img_proposal = proposals[i]

        prop_num = min(img_proposal.shape[0], proposal_nums[-1])
        if gts[i] is None or gts[i].shape[0] == 0:
            ious = np.zeros((0, img_proposal.shape[0]), dtype=np.float32)
        else:
            ious = bbox_overlaps(
                torch.tensor(gts[i]),
                torch.tensor(img_proposal[:prop_num, :4]))
            ious = ious.data.numpy()
        all_ious.append(ious)
    all_ious = np.array(all_ious)
    recalls = _recalls(all_ious, proposal_nums, iou_thrs)
    return recalls


def binary_precision_recall_curve(y_score, y_true):
    """Calculate the binary precision recall curve at step thresholds.

    Args:
        y_score (np.ndarray): Prediction scores for each class.
            Shape should be (num_classes, ).
        y_true (np.ndarray): Ground truth many-hot vector.
            Shape should be (num_classes, ).

    Returns:
        precision (np.ndarray): The precision of different thresholds.
        recall (np.ndarray): The recall of different thresholds.
        thresholds (np.ndarray): Different thresholds at which precison and
            recall are tested.
    """
    assert isinstance(y_score, np.ndarray)
    assert isinstance(y_true, np.ndarray)
    assert y_score.shape == y_true.shape

    # make y_true a boolean vector
    y_true = (y_true == 1)
    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind='mergesort')[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    # There may be ties in values, therefore find the `distinct_value_inds`
    distinct_value_inds = np.where(np.diff(y_score))[0]
    threshold_inds = np.r_[distinct_value_inds, y_true.size - 1]
    # accumulate the true positives with decreasing threshold
    tps = np.cumsum(y_true)[threshold_inds]
    fps = 1 + threshold_inds - tps
    thresholds = y_score[threshold_inds]

    precision = tps / (tps + fps)
    precision[np.isnan(precision)] = 0
    recall = tps / tps[-1]
    # stop when full recall attained
    # and reverse the outputs so recall is decreasing
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)

    return np.r_[precision[sl], 1], np.r_[recall[sl], 0], thresholds[sl]


def average_recall_at_avg_proposals(ground_truth,
                                    proposals,
                                    total_num_proposals,
                                    max_avg_proposals=None,
                                    temporal_iou_thresholds=np.linspace(
                                        0.5, 0.95, 10)):
    """Computes the average recall given an average number (percentile) of
    proposals per video.

    Args:
        ground_truth (dict): Dict containing the ground truth instances.
        proposals (dict): Dict containing the proposal instances.
        total_num_proposals (int): Total number of proposals in the
            proposal dict.
        max_avg_proposals (int | None): Max number of proposals for one video.
            Default: None.
        temporal_iou_thresholds (np.ndarray): 1D array with temporal_iou
            thresholds. Default: ``np.linspace(0.5, 0.95, 10)``.

    Returns:
        tuple([np.ndarray, np.ndarray, np.ndarray, float]):
            (recall, average_recall, proposals_per_video, auc)
            In recall, ``recall[i,j]`` is recall at i-th temporal_iou threshold
            at the j-th average number (percentile) of average number of
            proposals per video. The average_recall is recall averaged
            over a list of temporal_iou threshold (1D array). This is
            equivalent to ``recall.mean(axis=0)``. The ``proposals_per_video``
            is the average number of proposals per video. The auc is the area
            under ``AR@AN`` curve.
    """

    total_num_videos = len(ground_truth)

    if not max_avg_proposals:
        max_avg_proposals = float(total_num_proposals) / total_num_videos

    ratio = (max_avg_proposals * float(total_num_videos) / total_num_proposals)

    # For each video, compute temporal_iou scores among the retrieved proposals
    score_list = []
    total_num_retrieved_proposals = 0
    for video_id in ground_truth:
        # Get proposals for this video.
        proposals_video_id = proposals[video_id]
        this_video_proposals = proposals_video_id[:, :2]
        # Sort proposals by score.
        sort_idx = proposals_video_id[:, 2].argsort()[::-1]
        this_video_proposals = this_video_proposals[sort_idx, :].astype(
            np.float32)

        # Get ground-truth instances associated to this video.
        ground_truth_video_id = ground_truth[video_id]
        this_video_ground_truth = ground_truth_video_id[:, :2].astype(
            np.float32)
        if this_video_proposals.shape[0] == 0:
            n = this_video_ground_truth.shape[0]
            score_list.append(np.zeros((n, 1)))
            continue

        if this_video_proposals.ndim != 2:
            this_video_proposals = np.expand_dims(this_video_proposals, axis=0)
        if this_video_ground_truth.ndim != 2:
            this_video_ground_truth = np.expand_dims(
                this_video_ground_truth, axis=0)

        num_retrieved_proposals = np.minimum(
            int(this_video_proposals.shape[0] * ratio),
            this_video_proposals.shape[0])
        total_num_retrieved_proposals += num_retrieved_proposals
        this_video_proposals = this_video_proposals[:
                                                    num_retrieved_proposals, :]

        # Compute temporal_iou scores.
        t_iou = pairwise_temporal_iou(this_video_proposals,
                                      this_video_ground_truth)
        score_list.append(t_iou)

    # Given that the length of the videos is really varied, we
    # compute the number of proposals in terms of a ratio of the total
    # proposals retrieved, i.e. average recall at a percentage of proposals
    # retrieved per video.

    # Computes average recall.
    pcn_list = np.arange(1, 101) / 100.0 * (
        max_avg_proposals * float(total_num_videos) /
        total_num_retrieved_proposals)
    matches = np.empty((total_num_videos, pcn_list.shape[0]))
    positives = np.empty(total_num_videos)
    recall = np.empty((temporal_iou_thresholds.shape[0], pcn_list.shape[0]))
    # Iterates over each temporal_iou threshold.
    for ridx, temporal_iou in enumerate(temporal_iou_thresholds):
        # Inspect positives retrieved per video at different
        # number of proposals (percentage of the total retrieved).
        for i, score in enumerate(score_list):
            # Total positives per video.
            positives[i] = score.shape[0]
            # Find proposals that satisfies minimum temporal_iou threshold.
            true_positives_temporal_iou = score >= temporal_iou
            # Get number of proposals as a percentage of total retrieved.
            pcn_proposals = np.minimum(
                (score.shape[1] * pcn_list).astype(np.int), score.shape[1])

            for j, num_retrieved_proposals in enumerate(pcn_proposals):
                # Compute the number of matches
                # for each percentage of the proposals
                matches[i, j] = np.count_nonzero(
                    (true_positives_temporal_iou[:, :num_retrieved_proposals]
                     ).sum(axis=1))

        # Computes recall given the set of matches per video.
        recall[ridx, :] = matches.sum(axis=0) / positives.sum()

    # Recall is averaged.
    avg_recall = recall.mean(axis=0)

    # Get the average number of proposals per video.
    proposals_per_video = pcn_list * (
        float(total_num_retrieved_proposals) / total_num_videos)
    # Get AUC
    area_under_curve = np.trapz(avg_recall, proposals_per_video)
    auc = 100. * float(area_under_curve) / proposals_per_video[-1]
    return recall, avg_recall, proposals_per_video, auc


@METRICS.register_module()
class ARAN(BaseMetrics):

    def __init__(self,
                 logger,
                 max_avg_proposals=100,
                 temporal_iou_thresholds=np.linspace(0.5, 0.95, 10)):
        super().__init__(logger)

        if isinstance(temporal_iou_thresholds, list):
            temporal_iou_thresholds = np.array(temporal_iou_thresholds)

        self.max_avg_proposals = max_avg_proposals
        self.temporal_iou_thresholds = temporal_iou_thresholds

    def wrap_up(self, results):
        auc = results['auc']
        recall = results['recall']

        eval_results = {
            'auc': auc,
            'AR@1': np.mean(recall[:, 0]),
            'AR@5': np.mean(recall[:, 4]),
            'AR@10': np.mean(recall[:, 9]),
            'AR@100': np.mean(recall[:, 99])
        }
        return eval_results

    def __call__(self, proposal, ground_truth, kwargs=None):
        keys = ['recall', 'avg_recall', 'proposals_per_video', 'auc']
        num_proposals = kwargs['num_proposals']

        results = average_recall_at_avg_proposals(ground_truth, proposal,
                                                  num_proposals,
                                                  self.max_avg_proposals,
                                                  self.temporal_iou_thresholds)
        results = dict(zip(keys, results))
        eval_results = self.wrap_up(results)

        self.print_log_msg(eval_results)
        return eval_results
