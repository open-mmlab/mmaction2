import numpy as np


def temporal_iou(proposal_min, proposal_max, gt_min, gt_max):
    """Compute IoU score between a groundtruth bbox and the proposals.

    Args:
        proposal_min (list[float]): List of temporal anchor min.
        proposal_max (list[float]): List of temporal anchor max.
        gt_min (float): Groundtruth temporal box min.
        gt_max (float): Groundtruth temporal box max.

    Returns:
        list[float]: List of iou scores.
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
        list[float]: List of intersection over anchor scores.
    """
    len_anchors = np.array(proposal_max - proposal_min)
    int_tmin = np.maximum(proposal_min, gt_min)
    int_tmax = np.minimum(proposal_max, gt_max)
    inter_len = np.maximum(int_tmax - int_tmin, 0.)
    scores = np.divide(inter_len, len_anchors)
    return scores


def pairwise_temporal_iou(candidate_segments,
                          target_segments,
                          calculate_overlap_self=False):
    """Compute intersection over union between segments.

    Args:
        candidate_segments (np.ndarray): 1-dim/2-dim array in format
            ``[init, end]/[m x 2:=[init, end]]``.
        target_segments (np.ndarray): 2-dim array in format
            ``[n x 2:=[init, end]]``.
        calculate_overlap_self (bool): Whether to calculate overlap_self
            (union / candidate_length) or not. Default: False.

    Returns:
        t_iou (np.ndarray): 1-dim array [n] /
            2-dim array [n x m] with IoU ratio.
        t_overlap_self (np.ndarray, optional): 1-dim array [n] /
            2-dim array [n x m] with overlap_self, returns when
            calculate_overlap_self is True.
    """
    candidate_segments_ndim = candidate_segments.ndim
    if target_segments.ndim != 2 or candidate_segments_ndim not in [1, 2]:
        raise ValueError('Dimension of arguments is incorrect')

    if candidate_segments_ndim == 1:
        candidate_segments = candidate_segments[np.newaxis, :]

    n, m = target_segments.shape[0], candidate_segments.shape[0]
    t_iou = np.empty((n, m), dtype=np.float32)
    if calculate_overlap_self:
        t_overlap_self = np.empty((n, m), dtype=np.float32)

    for i in range(m):
        candidate_segment = candidate_segments[i, :]
        tt1 = np.maximum(candidate_segment[0], target_segments[:, 0])
        tt2 = np.minimum(candidate_segment[1], target_segments[:, 1])
        # Intersection including Non-negative overlap score.
        segments_intersection = (tt2 - tt1).clip(0)
        # Segment union.
        segments_union = ((target_segments[:, 1] - target_segments[:, 0]) +
                          (candidate_segment[1] - candidate_segment[0]) -
                          segments_intersection)
        # Compute overlap as the ratio of the intersection
        # over union of two segments.
        t_iou[:, i] = (segments_intersection.astype(float) / segments_union)
        if calculate_overlap_self:
            candidate_length = candidate_segment[1] - candidate_segment[0]
            t_overlap_self[:, i] = (
                segments_intersection.astype(float) / candidate_length)

    if candidate_segments_ndim == 1:
        t_iou = np.squeeze(t_iou, axis=1)
    if calculate_overlap_self:
        if candidate_segments_ndim == 1:
            t_overlap_self = np.squeeze(t_overlap_self, axis=1)
        return t_iou, t_overlap_self

    return t_iou
