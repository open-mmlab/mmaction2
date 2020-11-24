import numpy as np


def overlap2d(bboxes1, bboxes2):
    """Calculate the overlap between each bbox of bboxes1 and bboxes2.

    Args:
        bboxes1 (np.ndarray): shape (n, 4).
        bboxes2 (np.ndarray): shape (k, 4).

    Returns:
        np.ndarray: Overlap between the boxes pairs.
    """
    x_min = np.maximum(bboxes1[:, 0], bboxes2[:, 0])
    y_min = np.maximum(bboxes1[:, 1], bboxes2[:, 1])
    x_max = np.minimum(bboxes1[:, 2], bboxes2[:, 2])
    y_max = np.minimum(bboxes1[:, 3], bboxes2[:, 3])

    width = np.maximum(0, x_max - x_min)
    height = np.maximum(0, y_max - y_min)

    return width * height


def area2d(box):
    """Calculate bounding boxes area.

    Args:
        box (np.ndarray): Bounding boxes in [x1, y1, x2, y2] format.

    Returns:
        np.ndarray: Area for bounding boxes.
    """
    width = box[:, 2] - box[:, 0]
    height = box[:, 3] - box[:, 1]

    return width * height


def iou2d(bboxes1, bboxes2):
    """Calculate the IoUs between each bbox of bboxes1 and bboxes2.

    Args:
        bboxes1 (np.ndarray): shape (n, 4).
        bboxes2 (np.ndarray): shape (k, 4).

    Returns:
        np.ndarray: IoU between the boxes pairs.
    """
    if bboxes1.ndim == 1:
        bboxes1 = bboxes1[None, :]
    if bboxes2.ndim == 1:
        bboxes2 = bboxes2[None, :]

    assert len(bboxes2) == 1
    overlap = overlap2d(bboxes1, bboxes2)

    return overlap / (area2d(bboxes1) + area2d(bboxes2) - overlap)


def iou3d(bboxes1, bboxes2):
    """Calculate the IoU3d regardless of temporal overlap between two pairs of
    bboxes.

    Args:
        bboxes1 (np.ndarray): shape (n, 4).
        bboxes2 (np.ndarray): shape (k, 4).

    Returns:
        np.ndarray: IoU3d regardless of temporal overlap.
    """

    assert bboxes1.shape[0] == bboxes2.shape[0]
    assert np.all(bboxes1[:, 0] == bboxes2[:, 0])

    overlap = overlap2d(bboxes1[:, 1:5], bboxes2[:, 1:5])

    return np.mean(
        overlap /
        (area2d(bboxes1[:, 1:5]) + area2d(bboxes2[:, 1:5]) - overlap))


def spatio_temporal_iou3d(bboxes1, bboxes2, spatial_only=False):
    """Calculate the IoU3d between two pairs of bboxes.

    Args:
        bboxes1 (np.ndarray): shape (n, 4).
        bboxes2 (np.ndarray): shape (k, 4).
        spatial_only (bool): Whether to consider the temporal overlap.
            Default: False.

    Returns:
        np.ndarray: IoU3d for bboxes between two tubes.
    """
    tmin = max(bboxes1[0, 0], bboxes2[0, 0])
    tmax = min(bboxes1[-1, 0], bboxes2[-1, 0])

    if tmax < tmin:
        return 0.0

    temporal_inter = tmax - tmin + 1
    temporal_union = (
        max(bboxes1[-1, 0], bboxes2[-1, 0]) -
        min(bboxes1[0, 0], bboxes2[0, 0]) + 1)

    tube1 = bboxes1[int(np.where(
        bboxes1[:,
                0] == tmin)[0]):int(np.where(bboxes1[:, 0] == tmax)[0]) + 1, :]
    tube2 = bboxes2[int(np.where(
        bboxes2[:,
                0] == tmin)[0]):int(np.where(bboxes2[:, 0] == tmax)[0]) + 1, :]

    return iou3d(tube1, tube2) * (1. if spatial_only else temporal_inter /
                                  temporal_union)


def spatio_temporal_nms3d(tubes, overlap=0.5):
    """NMS processing for tubes in spatio and temporal dimension.

    Args:
        tubes (np.ndarray): Bounding boxes in tubes.
        overlap (float): Threshold of overlap for nms

    Returns:
        np.ndarray[int]: Index for Selected bboxes.
    """
    if not tubes:
        return np.array([], dtype=np.int32)

    indexes = np.argsort([tube[1] for tube in tubes])
    indices = np.zeros(indexes.size, dtype=np.int32)
    counter = 0

    while indexes.size > 0:
        i = indexes[-1]
        indices[counter] = i
        counter += 1
        ious = np.array([
            spatio_temporal_iou3d(tubes[index_list][0], tubes[i][0])
            for index_list in indexes[:-1]
        ])
        indexes = indexes[np.where(ious <= overlap)[0]]

    return indices[:counter]


def nms2d(boxes, overlap=0.6):
    """NMS processing.

    Args:
        boxes (np.ndarray): shape (n, 4).
        overlap (float): Threshold of overlap. Default: 0.6.

    Returns:
        np.ndarray: Result bboxes.
    """
    if boxes.size == 0:
        return np.array([], dtype=np.int32)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    scores = boxes[:, 4]
    areas = (x2 - x1) * (y2 - y1)
    order = np.argsort(scores)[::-1]
    weight = np.zeros_like(scores) + 1

    while order.size > 0:
        i = order[0]

        x_min = np.maximum(x1[i], x1[order[1:]])
        y_min = np.maximum(y1[i], y1[order[1:]])
        x_max = np.minimum(x2[i], x2[order[1:]])
        y_max = np.minimum(y2[i], y2[order[1:]])

        inter = np.maximum(0.0, x_max - x_min) * np.maximum(0.0, y_max - y_min)
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        index = np.where(iou > overlap)[0]
        weight[order[index + 1]] = 1 - iou[index]

        index2 = np.where(iou <= overlap)[0]
        order = order[index2 + 1]

    boxes[:, 4] = boxes[:, 4] * weight

    return boxes
