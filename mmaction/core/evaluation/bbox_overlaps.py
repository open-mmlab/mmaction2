import numpy as np


def overlap2d(origin_box, target_box):
    """Compute the overlaps between original boxes and target boxes.

    Args:
        origin_box (np.ndarray): Original bounding boxes.
        target_box (np.ndarray): Target bounding boxes.

    Returns:
        np.ndarray: Overlap between the boxes pairs.
    """
    x_min = np.maximum(origin_box[:, 0], target_box[:, 0])
    y_min = np.maximum(origin_box[:, 1], target_box[:, 1])
    x_max = np.minimum(origin_box[:, 2], target_box[:, 2])
    y_max = np.minimum(origin_box[:, 3], target_box[:, 3])

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


def iou2d(origin_box, target_box):
    """Compute the IoU between original boxes and target boxes.

    Args:
        origin_box (np.ndarray): Original bounding boxes.
        target_box (np.ndarray): Target bounding boxes.

    Returns:
        np.ndarray: IoU between the boxes pairs.
    """
    if origin_box.ndim == 1:
        origin_box = origin_box[None, :]
    if target_box.ndim == 1:
        target_box = target_box[None, :]

    assert len(target_box) == 1
    overlap = overlap2d(origin_box, target_box)

    return overlap / (area2d(origin_box) + area2d(target_box) - overlap)


def iou3d(origin_box, target_box):

    assert origin_box.shape[0] == target_box.shape[0]
    assert np.all(origin_box[:, 0] == target_box[:, 0])

    overlap = overlap2d(origin_box[:, 1:5], target_box[:, 1:5])

    return np.mean(
        overlap /
        (area2d(origin_box[:, 1:5]) + area2d(target_box[:, 1:5]) - overlap))


def spatio_temporal_iou3d(origin_box, target_box, spatial_only=False):
    tmin = max(origin_box[0, 0], target_box[0, 0])
    tmax = min(origin_box[-1, 0], target_box[-1, 0])

    if tmax < tmin:
        return 0.0

    temporal_inter = tmax - tmin + 1
    temporal_union = (
        max(origin_box[-1, 0], target_box[-1, 0]) -
        min(origin_box[0, 0], target_box[0, 0]) + 1)

    tube1 = origin_box[int(np.where(
        origin_box[:,
                   0] == tmin)[0]):int(np.where(origin_box[:, 0] == tmax)[0]) +
                       1, :]
    tube2 = target_box[int(np.where(
        target_box[:,
                   0] == tmin)[0]):int(np.where(target_box[:, 0] == tmax)[0]) +
                       1, :]

    return iou3d(tube1, tube2) * (1. if spatial_only else temporal_inter /
                                  temporal_union)


def spatio_temporal_nms3d(tubes, overlap=0.5):
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
