from detectron2.layers import batched_nms


def ml_nms(boxlist,
           nms_thresh,
           max_proposals=-1,
           score_field='scores',
           label_field='labels'):
    """Performs non-maximum suppression on a boxlist, with scores specified in
    a boxlist field via score_field.

    Arguments:
        boxlist(BoxList)
        nms_thresh (float)
        max_proposals (int): if > 0, then only the top max_proposals are kept
            after non-maximum suppression
        score_field (str)
    """
    if nms_thresh <= 0:
        return boxlist
    if boxlist.has('pred_boxes'):
        boxes = boxlist.pred_boxes.tensor
        labels = boxlist.pred_classes
    else:
        boxes = boxlist.proposal_boxes.tensor
        labels = boxlist.proposal_boxes.tensor.new_zeros(
            len(boxlist.proposal_boxes.tensor))
    scores = boxlist.scores

    keep = batched_nms(boxes, scores, labels, nms_thresh)
    if max_proposals > 0:
        keep = keep[:max_proposals]
    boxlist = boxlist[keep]
    return boxlist
