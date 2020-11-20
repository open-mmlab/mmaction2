import numpy as np
import torch


def bbox2roi(bbox_list):
    """Convert a list of bboxes to roi format.

    Args:
        bbox_list (list[Tensor]): a list of bboxes
        corresponding to a batch of images

    Returns:
        Tensor: shape (n, 5), [batch_ind, x1, y1, x2, y2]
    """
    rois_list = []
    for img_id, bboxes in enumerate(bbox_list):
        if bboxes.size(0) > 0:
            img_inds = bboxes.new_full((bboxes.size(0), 1), img_id)
            rois = torch.cat([img_inds, bboxes[:, :4]], dim=-1)
        else:
            rois = bboxes.new_zeros((0, 5))
        rois_list.append(rois)
    rois = torch.cat(rois_list, 0)
    return rois


def bbox2result(bboxes, labels, num_classes, thr=0.01):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (Tensor): shape (n, 4)
        labels (Tensor): shape (n, #num_classes)
        num_classes (int): class number, including background class
    Returns:
        list(ndarray): bbox results of each class
    """
    if bboxes.shape[0] == 0:
        return [
            np.zeros((0, 5), dtype=np.float32) for i in range(num_classes - 1)
        ]
    else:
        bboxes = bboxes.cpu().numpy()
        labels = labels.cpu().numpy()

        # We only handle multilabel now
        assert labels.shape[-1] > 1

        scores = labels  # rename for clarification
        thr = (thr, ) * num_classes if isinstance(thr, float) else thr
        assert scores.shape[1] == num_classes
        assert len(thr) == num_classes

        result = []
        for i in range(num_classes - 1):
            where = scores[:, i + 1] > thr[i + 1]
            result.append(
                np.concatenate((bboxes[where, :4], scores[where, i + 1:i + 2]),
                               axis=1))
        return result
