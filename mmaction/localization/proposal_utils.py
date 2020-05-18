import numpy as np


def temporal_iou(proposal_min, proposal_max, gt_min, gt_max):
    """Compute IoU score between a groundtruth bbox and the proposals.

    Args:
        proposal_min (list[float]): List of temporal anchor min.
        proposal_max (list[float]): List of temporal anchor max.
        gt_min (float): Groundtruth temporal box min.
        gt_max (float): Groundtruth temporal box max.

    Returns:
        jaccard (list[float]): List of iou scores.
    """
    len_anchors = proposal_max - proposal_min
    int_tmin = np.maximum(proposal_min, gt_min)
    int_tmax = np.minimum(proposal_max, gt_max)
    inter_len = np.maximum(int_tmax - int_tmin, 0.)
    union_len = len_anchors - inter_len + gt_max - gt_min
    jaccard = np.divide(inter_len, union_len)
    return jaccard


def temporal_iop(proposal_min, proposal_max, gt_min, gt_max):
    """Compute IoP score between a groundtruth bbox and the proposals.

    Compute the IoP which is defined as the overlap ratio with
    groundtruth proportional to the duration of this proposal.

    Args:
        proposal_min (list[float]): List of temporal anchor min.
        proposal_max (list[float]): List of temporal anchor max.
        gt_min (float): Groundtruth temporal box min.
        gt_max (float): Groundtruth temporal box max.

    Returns:
        scores (list[float]): List of intersection over anchor scores.
    """
    len_anchors = proposal_max - proposal_min
    int_tmin = np.maximum(proposal_min, gt_min)
    int_tmax = np.minimum(proposal_max, gt_max)
    inter_len = np.maximum(int_tmax - int_tmin, 0.)
    scores = np.divide(inter_len, len_anchors)
    return scores
