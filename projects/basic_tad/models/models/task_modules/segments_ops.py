from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from mmcv.ops import batched_nms
from terminaltables import AsciiTable
from mmengine.logging import print_log


def convert_1d_to_2d_bboxes(bboxes_1d, fixed_dim_value=0):
    """
    Convert 1D bounding boxes to pseudo 2D bounding boxes by adding a fixed dimension.

    Args:
        bboxes_1d (torch.Tensor): 1D bounding boxes tensor of shape (N, 2)
        fixed_dim_value (float): Value to set for the fixed dimension in the 2D bounding boxes

    Returns:
        torch.Tensor: Pseudo 2D bounding boxes tensor of shape (N, 4)
    """
    # Get the number of bounding boxes
    num_bboxes = bboxes_1d.shape[0]

    # Initialize the 2D bounding boxes tensor
    bboxes_2d = torch.zeros((num_bboxes, 4), device=bboxes_1d.device, dtype=bboxes_1d.dtype)

    # Set the fixed dimension value for ymin and ymax
    bboxes_2d[:, 1] = fixed_dim_value
    bboxes_2d[:, 3] = fixed_dim_value + 1

    # Copy the 1D intervals (xmin and xmax) to the 2D bounding boxes
    bboxes_2d[:, 0::2] = bboxes_1d

    return bboxes_2d


def convert_2d_to_1d_bboxes(bboxes_2d):
    """
    Convert pseudo 2D bounding boxes back to 1D bounding boxes by extracting xmin and xmax.

    Args:
        bboxes_2d (torch.Tensor): Pseudo 2D bounding boxes tensor of shape (N, 4)

    Returns:
        torch.Tensor: 1D bounding boxes tensor of shape (N, 2)
    """
    # Extract xmin and xmax (first and third columns) from the 2D bounding boxes
    bboxes_1d = bboxes_2d[:, 0::2]

    return bboxes_1d


def batched_nms1d(bboxes_1d, *args, **kwargs):
    """
    Apply Non-Maximum Suppression (NMS) on 1D bounding boxes by converting them to pseudo 2D bounding boxes,
    using a 2D NMS function, and converting the results back to 1D bounding boxes.

    Args:
        bboxes_1d (torch.Tensor): 1D bounding boxes tensor of shape (N, 2)
        *args: Additional arguments to pass to the batched_nms function
        **kwargs: Additional keyword arguments to pass to the batched_nms function

    Returns:
        tuple: A tuple containing:
            - torch.Tensor: The 1D bounding boxes after NMS, with shape (N', 2)
            - torch.Tensor: The indices of the kept bounding boxes, with shape (N',)
    """
    bboxes_2d = convert_1d_to_2d_bboxes(bboxes_1d)
    boxes, keep = batched_nms(bboxes_2d, *args, **kwargs)
    return convert_2d_to_1d_bboxes(boxes), keep


# Below all adapted from basicTAD
def segment_overlaps(segments1,
                     segments2,
                     mode='iou',
                     is_aligned=False,
                     eps=1e-6,
                     detect_overlap_edge=False):
    """Calculate overlap between two set of segments.
    If ``is_aligned`` is ``False``, then calculate the ious between each
    segment of segments1 and segments2, otherwise the ious between each aligned
     pair of segments1 and segments2.
    Args:
        segments1 (Tensor): shape (m, 2) in <t1, t2> format or empty.
        segments2 (Tensor): shape (n, 2) in <t1, t2> format or empty.
            If is_aligned is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union) or iof (intersection over
            foreground).
    Returns:
        ious(Tensor): shape (m, n) if is_aligned == False else shape (m, 1)
    Example:
        >>> segments1 = torch.FloatTensor([
        >>>     [0, 10],
        >>>     [10, 20],
        >>>     [32, 38],
        >>> ])
        >>> segments2 = torch.FloatTensor([
        >>>     [0, 20],
        >>>     [0, 19],
        >>>     [10, 20],
        >>> ])
        >>> segment_overlaps(segments1, segments2)
        tensor([[0.5000, 0.5263, 0.0000],
                [0.0000, 0.4500, 1.0000],
                [0.0000, 0.0000, 0.0000]])
    Example:
        >>> empty = torch.FloatTensor([])
        >>> nonempty = torch.FloatTensor([
        >>>     [0, 9],
        >>> ])
        >>> assert tuple(segment_overlaps(empty, nonempty).shape) == (0, 1)
        >>> assert tuple(segment_overlaps(nonempty, empty).shape) == (1, 0)
        >>> assert tuple(segment_overlaps(empty, empty).shape) == (0, 0)
    """

    is_numpy = False
    if isinstance(segments1, np.ndarray):
        segments1 = torch.from_numpy(segments1)
        is_numpy = True
    if isinstance(segments2, np.ndarray):
        segments2 = torch.from_numpy(segments2)
        is_numpy = True

    segments1, segments2 = segments1.float(), segments2.float()

    assert mode in ['iou', 'iof']
    # Either the segments are empty or the length of segments's last dimenstion
    # is 2
    assert (segments1.size(-1) == 2 or segments1.size(0) == 0)
    assert (segments2.size(-1) == 2 or segments2.size(0) == 0)

    rows = segments1.size(0)
    cols = segments2.size(0)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        return segments1.new(rows, 1) if is_aligned else segments2.new(
            rows, cols)

    if is_aligned:
        start = torch.max(segments1[:, 0], segments2[:, 0])  # [rows]
        end = torch.min(segments1[:, 1], segments2[:, 1])  # [rows]

        overlap = end - start
        if detect_overlap_edge:
            overlap[overlap == 0] += eps
        overlap = overlap.clamp(min=0)  # [rows, 2]
        area1 = segments1[:, 1] - segments1[:, 0]

        if mode == 'iou':
            area2 = segments2[:, 1] - segments2[:, 0]
            union = area1 + area2 - overlap
        else:
            union = area1
    else:
        start = torch.max(segments1[:, None, 0], segments2[:,
                                                 0])  # [rows, cols]
        end = torch.min(segments1[:, None, 1], segments2[:, 1])  # [rows, cols]

        overlap = end - start
        if detect_overlap_edge:
            overlap[overlap == 0] += eps
        overlap = overlap.clamp(min=0)  # [rows, 2]
        area1 = segments1[:, 1] - segments1[:, 0]

        if mode == 'iou':
            area2 = segments2[:, 1] - segments2[:, 0]
            union = area1[:, None] + area2 - overlap
        else:
            union = area1[:, None]

    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union

    if is_numpy:
        ious = ious.numpy()

    return ious


# apart from basicTAD
def anchor_inside_flags(flat_anchors, valid_flags, tsize, allowed_border=0):
    """Check whether the anchors are inside the border.

    Args:
        flat_anchors (torch.Tensor): Flatten anchors, shape (n, 2).
        valid_flags (torch.Tensor): An existing valid flags of anchors.
        tsize (int): Temporal size of current video.
        allowed_border (int, optional): The border to allow the valid anchor.
            Defaults to 0.

    Returns:
        torch.Tensor: Flags indicating whether the anchors are inside a
            valid range.
    """
    if allowed_border >= 0:
        inside_flags = (
                valid_flags & (flat_anchors[:, 0] >= -allowed_border) &
                (flat_anchors[:, 1] < tsize + allowed_border))
    else:
        inside_flags = valid_flags
    return inside_flags


def average_precision(recalls, precisions, mode='area'):
    """Calculate average precision (for single or multiple scales).

    Args:
        recalls (ndarray): shape (num_scales, num_dets) or (num_dets, )
        precisions (ndarray): shape (num_scales, num_dets) or (num_dets, )
        mode (str): 'area' or '11points', 'area' means calculating the area
            under precision-recall curve, '11points' means calculating
            the average precision of recalls at [0, 0.1, ..., 1]

    Returns:
        float or ndarray: calculated average precision
    """
    no_scale = False
    if recalls.ndim == 1:
        no_scale = True
        recalls = recalls[np.newaxis, :]
        precisions = precisions[np.newaxis, :]
    assert recalls.shape == precisions.shape and recalls.ndim == 2
    num_scales = recalls.shape[0]
    ap = np.zeros(num_scales, dtype=np.float32)
    if mode == 'area':
        zeros = np.zeros((num_scales, 1), dtype=recalls.dtype)
        ones = np.ones((num_scales, 1), dtype=recalls.dtype)
        mrec = np.hstack((zeros, recalls, ones))
        mpre = np.hstack((zeros, precisions, zeros))
        for i in range(mpre.shape[1] - 1, 0, -1):
            mpre[:, i - 1] = np.maximum(mpre[:, i - 1], mpre[:, i])
        for i in range(num_scales):
            ind = np.where(mrec[i, 1:] != mrec[i, :-1])[0]
            ap[i] = np.sum(
                (mrec[i, ind + 1] - mrec[i, ind]) * mpre[i, ind + 1])
    elif mode == '11points':
        for i in range(num_scales):
            for thr in np.arange(0, 1 + 1e-3, 0.1):
                precs = precisions[i, recalls[i, :] >= thr]
                prec = precs.max() if precs.size > 0 else 0
                ap[i] += prec
            ap /= 11
    else:
        raise ValueError(
            'Unrecognized mode, only "area" and "11points" are supported')
    if no_scale:
        ap = ap[0]
    return ap


def tpfp_default(det_segments,
                 gt_segments,
                 gt_segments_ignore=None,
                 iou_thr=0.5,
                 area_ranges=None):
    """Check if detected segments are true positive or false positive.

    Args:
        det_segments (ndarray): Detected segments of this video, of shape
            (m, 3).
        gt_segments (ndarray): GT segments of this video, of shape (n, 2).
        gt_segments_ignore (ndarray): Ignored gt segments of this video,
            of shape (k, 2). Default: None
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        area_ranges (list[tuple] | None): Range of segment areas to be
            evaluated, in the format [(min1, max1), (min2, max2), ...].
            Default: None.

    Returns:
        tuple[np.ndarray]: (tp, fp) whose elements are 0 and 1. The shape of
            each array is (num_scales, m).
    """
    # an indicator of ignored gts
    gt_ignore_inds = np.concatenate(
        (np.zeros(gt_segments.shape[0], dtype=np.bool),
         np.ones(gt_segments_ignore.shape[0], dtype=np.bool)))
    # stack gt_segments and gt_segments_ignore for convenience
    gt_segments = np.vstack((gt_segments, gt_segments_ignore))

    num_dets = det_segments.shape[0]
    num_gts = gt_segments.shape[0]
    if area_ranges is None:
        area_ranges = [(None, None)]
    num_scales = len(area_ranges)
    # tp and fp are of shape (num_scales, num_gts), each row is tp or fp of
    # a certain scale
    tp = np.zeros((num_scales, num_dets), dtype=np.float32)
    fp = np.zeros((num_scales, num_dets), dtype=np.float32)

    # if there is no gt segments in this video, then all det segments
    # within area range are false positives
    if gt_segments.shape[0] == 0:
        if area_ranges == [(None, None)]:
            fp[...] = 1
        else:
            det_areas = det_segments[:, 1] - det_segments[:, 0]
            for i, (min_area, max_area) in enumerate(area_ranges):
                fp[i, (det_areas >= min_area) & (det_areas < max_area)] = 1
        return tp, fp

    # if there is no det segments in this video, return tp, fp
    if det_segments.shape[0] == 0:
        return tp, fp

    ious = segment_overlaps(det_segments[:, :2], gt_segments)

    # for each det, the max iou with all gts
    ious_max = ious.max(axis=1)
    # for each det, which gt overlaps most with it
    ious_argmax = ious.argmax(axis=1)
    # sort all dets in descending order by scores
    sort_inds = np.argsort(-det_segments[:, -1])
    for k, (min_area, max_area) in enumerate(area_ranges):
        gt_covered = np.zeros(num_gts, dtype=bool)
        # if no area range is specified, gt_area_ignore is all False
        if min_area is None:
            gt_area_ignore = np.zeros_like(gt_ignore_inds, dtype=bool)
        else:
            gt_areas = gt_segments[:, 1] - gt_segments[:, 0]
            gt_area_ignore = (gt_areas < min_area) | (gt_areas >= max_area)
        for i in sort_inds:
            if ious_max[i] >= iou_thr:
                matched_gt = ious_argmax[i]
                if not (gt_ignore_inds[matched_gt]
                        or gt_area_ignore[matched_gt]):
                    if not gt_covered[matched_gt]:
                        gt_covered[matched_gt] = True
                        tp[k, i] = 1
                    else:
                        fp[k, i] = 1
                # otherwise ignore this detected segment, tp = 0, fp = 0
            elif min_area is None:
                fp[k, i] = 1
            else:
                segment = det_segments[i, :2]
                area = segment[1] - segment[0]
                if min_area <= area < max_area:
                    fp[k, i] = 1
    return tp, fp


def tpfp_anet(det_segments,
              gt_segments,
              gt_segments_ignore=None,
              iou_thr=0.5,
              scale_ranges=None):
    """Check if detected segments are true positive or false positive.

    Args:
        det_segments (ndarray): Detected segments of this video, of shape
            (m, 3).
        gt_segments (ndarray): GT segments of this video, of shape (n, 2).
        gt_segments_ignore (ndarray): Ignored gt segments of this video,
            of shape (k, 2). Default: None
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        scale_ranges (list[tuple] | None): Range of segment areas to be
            evaluated, in the format [(min1, max1), (min2, max2), ...].
            Default: None.

    Returns:
        tuple[np.ndarray]: (tp, fp) whose elements are 0 and 1. The shape of
            each array is (num_scales, m).
    """
    assert gt_segments_ignore.shape[0] == 0, (
        'gt_segments_ignore shape should be 0.')
    assert scale_ranges is None, 'scale_ranges should set to None.'

    num_dets = det_segments.shape[0]
    num_gts = gt_segments.shape[0]
    num_scales = 1
    # tp and fp are of shape (num_scales, num_dets), each row is tp or fp of
    # a certain scale
    tp = np.zeros((num_scales, num_dets), dtype=np.float32)
    fp = np.zeros((num_scales, num_dets), dtype=np.float32)

    # if there is no gt segments in this video, then all det segments
    # within area range are false positives
    if gt_segments.shape[0] == 0:
        fp[...] = 1
        return tp, fp

    ious = segment_overlaps(det_segments[:, :2], gt_segments)

    # sort all dets in descending order by scores
    sort_inds = np.argsort(-det_segments[:, -1])
    gt_covered = np.zeros(num_gts, dtype=bool)
    for i in sort_inds:
        gt_ious = ious[i]
        sort_gt_inds = np.argsort(-gt_ious)
        for matched_gt in sort_gt_inds:
            if gt_ious[matched_gt] >= iou_thr:
                if not gt_covered[matched_gt]:
                    gt_covered[matched_gt] = True
                    tp[0, i] = 1
                    break
            else:
                fp[0, i] = 1
        if fp[0, i] == 0 and tp[0, i] == 0:
            fp[0, i] = 1
    return tp, fp


def get_cls_results(det_results, annotations, class_id):
    """Get det results and gt information of a certain class.

    Args:
        det_results (list[list]): Same as `eval_map()`.
        annotations (list[dict]): Same as `eval_map()`.
        class_id (int): ID of a specific class.

    Returns:
        tuple[list[np.ndarray]]: detected segments, gt segments,
            ignored gt segments
    """
    cls_dets = [video_res[class_id] for video_res in det_results]
    cls_gts = []
    cls_gts_ignore = []
    for ann in annotations:
        gt_inds = ann['labels'] == class_id
        cls_gts.append(ann['segments'][gt_inds, :])

        if ann.get('labels_ignore', None) is not None:
            ignore_inds = ann['labels_ignore'] == class_id
            cls_gts_ignore.append(ann['segments_ignore'][ignore_inds, :])
        else:
            cls_gts_ignore.append(np.empty((0, 2), dtype=np.float32))

    return cls_dets, cls_gts, cls_gts_ignore


def eval_map(det_results,
             annotations,
             scale_ranges=None,
             iou_thr=0.5,
             mode=None,
             logger=None,
             nproc=4,
             label_names=None):
    """Evaluate mAP of a dataset.

    Args:
        det_results (list[list]): [[cls1_det, cls2_det, ...], ...].
            The outer list indicates videos, and the inner list indicates
            per-class detected segments.
        annotations (list[dict]): Ground truth annotations where each item of
            the list indicates a video. Keys of annotations are:

            - `segments`: numpy array of shape (n, 2)
            - `labels`: numpy array of shape (n, )
            - `segments_ignore` (optional): numpy array of shape (k, 2)
            - `labels_ignore` (optional): numpy array of shape (k, )
        scale_ranges (list[tuple] | None): Range of scales to be evaluated,
            in the format [(min1, max1), (min2, max2), ...]. A range of
            (32, 64) means the area range between (32, 64).
            Default: None.
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        mode (str | None): Mode name, there are minor differences in metrics
            for different modes, e.g. "anet", "voc07", "voc12" etc.
            Default: None.
        logger (logging.Logger | str | None): The way to print the mAP
            summary. See `mmdet.utils.print_log()` for details. Default: None.
        nproc (int): Processes used for computing TP and FP.
            Default: 4.
        label_names (list[str] | None): Label names.

    Returns:
        tuple: (mAP, [dict, dict, ...])
    """
    assert len(det_results) == len(annotations)

    num_videos = len(det_results)
    num_scales = len(scale_ranges) if scale_ranges is not None else 1
    num_classes = len(det_results[0])  # positive class num

    eval_results = []
    for i in range(num_classes):
        # get gt and det segments of this class
        cls_dets, cls_gts, cls_gts_ignore = get_cls_results(
            det_results, annotations, i)

        # choose proper function according to datasets to compute tp and fp
        if mode in ['anet']:
            tpfp_func = tpfp_anet
        else:
            tpfp_func = tpfp_default
        # compute tp and fp for each video with multiple processes
        with ThreadPoolExecutor(nproc) as executor:
            tpfp = executor.map(tpfp_func, cls_dets, cls_gts, cls_gts_ignore,
                                [iou_thr for _ in range(num_videos)],
                                [scale_ranges for _ in range(num_videos)])
        tp, fp = tuple(zip(*tpfp))
        # calculate gt number of each scale
        # ignored gts or gts beyond the specific scale are not counted
        num_gts = np.zeros(num_scales, dtype=int)
        for j, segment in enumerate(cls_gts):
            if scale_ranges is None:
                num_gts[0] += segment.shape[0]
            else:
                gt_areas = segment[:, 1] - segment[:, 0]
                for k, (min_area, max_area) in enumerate(scale_ranges):
                    num_gts[k] += np.sum((gt_areas >= min_area)
                                         & (gt_areas < max_area))
        # sort all det segments by score, also sort tp and fp
        cls_dets = np.vstack(cls_dets)
        num_dets = cls_dets.shape[0]
        sort_inds = np.argsort(-cls_dets[:, -1])
        tp = np.hstack(tp)[:, sort_inds]
        fp = np.hstack(fp)[:, sort_inds]
        # calculate recall and precision with tp and fp
        tp = np.cumsum(tp, axis=1)
        fp = np.cumsum(fp, axis=1)
        eps = np.finfo(np.float32).eps
        recalls = tp / np.maximum(num_gts[:, np.newaxis], eps)
        precisions = tp / np.maximum((tp + fp), eps)
        # calculate AP
        if scale_ranges is None:
            recalls = recalls[0, :]
            precisions = precisions[0, :]
            num_gts = num_gts.item()
        ap_mode = 'area' if mode != 'voc07' else '11points'
        ap = average_precision(recalls, precisions, ap_mode)
        eval_results.append({
            'num_gts': num_gts,
            'num_dets': num_dets,
            'recall': recalls,
            'precision': precisions,
            'ap': ap
        })
    if scale_ranges is not None:
        # shape (num_classes, num_scales)
        all_ap = np.vstack([cls_result['ap'] for cls_result in eval_results])
        all_num_gts = np.vstack(
            [cls_result['num_gts'] for cls_result in eval_results])
        mean_ap = []
        for i in range(num_scales):
            if np.any(all_num_gts[:, i] > 0):
                mean_ap.append(all_ap[all_num_gts[:, i] > 0, i].mean())
            else:
                mean_ap.append(0.0)
    else:
        aps = []
        for cls_result in eval_results:
            if cls_result['num_gts'] > 0:
                aps.append(cls_result['ap'])
        mean_ap = np.array(aps).mean().item() if aps else 0.0

    print_map_summary(
        mean_ap, eval_results, label_names, scale_ranges, logger=logger)

    return mean_ap, eval_results


def print_map_summary(mean_ap,
                      results,
                      label_names=None,
                      scale_ranges=None,
                      logger=None):
    """Print mAP and results of each class.

    A table will be printed to show the gts/dets/recall/AP of each class and
    the mAP.

    Args:
        mean_ap (float): Calculated from `eval_map()`.
        results (list[dict]): Calculated from `eval_map()`.
        label_names (list[str] | None): Label names.
        scale_ranges (list[tuple] | None): Range of scales to be evaluated.
        logger (logging.Logger | str | None): The way to print the mAP
            summary. See `mmdet.utils.print_log()` for details. Default: None.
    """

    if logger == 'silent':
        return

    if isinstance(results[0]['ap'], np.ndarray):
        num_scales = len(results[0]['ap'])
    else:
        num_scales = 1

    if scale_ranges is not None:
        assert len(scale_ranges) == num_scales

    num_classes = len(results)

    recalls = np.zeros((num_scales, num_classes), dtype=np.float32)
    aps = np.zeros((num_scales, num_classes), dtype=np.float32)
    num_gts = np.zeros((num_scales, num_classes), dtype=int)
    for i, cls_result in enumerate(results):
        if cls_result['recall'].size > 0:
            recalls[:, i] = np.array(cls_result['recall'], ndmin=2)[:, -1]
        aps[:, i] = cls_result['ap']
        num_gts[:, i] = cls_result['num_gts']

    if label_names is None:
        label_names = [str(i) for i in range(num_classes)]

    if not isinstance(mean_ap, list):
        mean_ap = [mean_ap]

    header = ['class', 'gts', 'dets', 'recall', 'ap']
    for i in range(num_scales):
        if scale_ranges is not None:
            print_log(f'Scale range {scale_ranges[i]}', logger=logger)
        table_data = [header]
        for j in range(num_classes):
            row_data = [
                label_names[j], num_gts[i, j], results[j]['num_dets'],
                f'{recalls[i, j]:.3f}', f'{aps[i, j]:.3f}'
            ]
            table_data.append(row_data)
        table_data.append(['mAP', '', '', '', f'{mean_ap[i]:.3f}'])
        table = AsciiTable(table_data)
        table.inner_footing_row_border = True
        print_log('\n' + table.table, logger=logger)
