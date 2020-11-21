import torch


def bbox_target(pos_bboxes_list,
                neg_bboxes_list,
                pos_gt_labels_list,
                cfg,
                concat=True):
    labels, label_weights = [], []
    for pos_bboxes, neg_bboxes, neg_gt_bboxes in zip(pos_bboxes_list,
                                                     neg_bboxes_list,
                                                     pos_gt_labels_list):
        label, label_weight = bbox_target_single(
            pos_bboxes, neg_bboxes, neg_gt_bboxes, cfg=cfg)
        labels.append(label)
        label_weights.append(label_weight)

    if concat:
        labels = torch.cat(labels, 0)
        label_weights = torch.cat(label_weights, 0)
    return labels, label_weights


# The class_weight is used for the classification loss, default 1
# The pos_weight is used for the positive bboxes (if train detection),
# default -1
def bbox_target_single(pos_bboxes, neg_bboxes, pos_gt_labels, cfg):
    num_pos = pos_bboxes.size(0)
    num_neg = neg_bboxes.size(0)
    num_samples = num_pos + num_neg

    label_len = len(pos_gt_labels[0])

    assert label_len > 1
    labels = pos_bboxes.new_zeros(num_samples, label_len)

    label_weights = pos_bboxes.new_zeros(num_samples)

    if num_pos > 0:
        labels[:num_pos] = pos_gt_labels
        pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
        label_weights[:num_pos] = pos_weight
    if num_neg > 0:
        label_weights[-num_neg:] = 1.0

    return labels, label_weights
