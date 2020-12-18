import os.path as osp
import random

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from mmaction.core import (ActivityNetDetection,
                           average_recall_at_avg_proposals, confusion_matrix,
                           get_weighted_score, mean_average_precision,
                           mean_class_accuracy, mmit_mean_average_precision,
                           pairwise_temporal_iou, top_k_accuracy)


def gt_confusion_matrix(gt_labels, pred_labels, normalize=None):
    """Calculate the ground truth confusion matrix."""
    max_index = max(max(gt_labels), max(pred_labels))
    confusion_mat = np.zeros((max_index + 1, max_index + 1), dtype=np.int64)
    for gt, pred in zip(gt_labels, pred_labels):
        confusion_mat[gt][pred] += 1
    del_index = []
    for i in range(max_index):
        if sum(confusion_mat[i]) == 0 and sum(confusion_mat[:, i]) == 0:
            del_index.append(i)
    confusion_mat = np.delete(confusion_mat, del_index, axis=0)
    confusion_mat = np.delete(confusion_mat, del_index, axis=1)

    if normalize is not None:
        confusion_mat = np.array(confusion_mat, dtype=np.float)
    m, n = confusion_mat.shape
    if normalize == 'true':
        for i in range(m):
            s = np.sum(confusion_mat[i], dtype=float)
            if s == 0:
                continue
            confusion_mat[i, :] = confusion_mat[i, :] / s
            print(confusion_mat[i, :])
    elif normalize == 'pred':
        for i in range(n):
            s = sum(confusion_mat[:, i])
            if s == 0:
                continue
            confusion_mat[:, i] = confusion_mat[:, i] / s
    elif normalize == 'all':
        s = np.sum(confusion_mat)
        if s != 0:
            confusion_mat /= s

    return confusion_mat


def test_activitynet_detection():
    data_prefix = osp.join(osp.dirname(__file__), 'data/test_eval_detection')
    gt_path = osp.join(data_prefix, 'gt.json')
    result_path = osp.join(data_prefix, 'result.json')
    detection = ActivityNetDetection(gt_path, result_path)

    results = detection.evaluate()
    mAP = np.array([
        0.71428571, 0.71428571, 0.71428571, 0.6875, 0.6875, 0.59722222,
        0.52083333, 0.52083333, 0.52083333, 0.5
    ])
    average_mAP = 0.6177579365079365

    assert_array_almost_equal(results[0], mAP)
    assert_array_almost_equal(results[1], average_mAP)


def test_confusion_matrix():
    # custom confusion_matrix
    gt_labels = [np.int64(random.randint(0, 9)) for _ in range(100)]
    pred_labels = np.random.randint(10, size=100, dtype=np.int64)

    for normalize in [None, 'true', 'pred', 'all']:
        cf_mat = confusion_matrix(pred_labels, gt_labels, normalize)
        gt_cf_mat = gt_confusion_matrix(gt_labels, pred_labels, normalize)
        assert_array_equal(cf_mat, gt_cf_mat)

    with pytest.raises(ValueError):
        # normalize must be in ['true', 'pred', 'all', None]
        confusion_matrix([1], [1], 'unsupport')

    with pytest.raises(TypeError):
        # y_pred must be list or np.ndarray
        confusion_matrix(0.5, [1])

    with pytest.raises(TypeError):
        # y_real must be list or np.ndarray
        confusion_matrix([1], 0.5)

    with pytest.raises(TypeError):
        # y_pred dtype must be np.int64
        confusion_matrix([0.5], [1])

    with pytest.raises(TypeError):
        # y_real dtype must be np.int64
        confusion_matrix([1], [0.5])


def test_topk():
    scores = [
        np.array([-0.2203, -0.7538, 1.8789, 0.4451, -0.2526]),
        np.array([-0.0413, 0.6366, 1.1155, 0.3484, 0.0395]),
        np.array([0.0365, 0.5158, 1.1067, -0.9276, -0.2124]),
        np.array([0.6232, 0.9912, -0.8562, 0.0148, 1.6413])
    ]

    # top1 acc
    k = (1, )
    top1_labels_0 = [3, 1, 1, 1]
    top1_labels_25 = [2, 0, 4, 3]
    top1_labels_50 = [2, 2, 3, 1]
    top1_labels_75 = [2, 2, 2, 3]
    top1_labels_100 = [2, 2, 2, 4]
    res = top_k_accuracy(scores, top1_labels_0, k)
    assert res == [0]
    res = top_k_accuracy(scores, top1_labels_25, k)
    assert res == [0.25]
    res = top_k_accuracy(scores, top1_labels_50, k)
    assert res == [0.5]
    res = top_k_accuracy(scores, top1_labels_75, k)
    assert res == [0.75]
    res = top_k_accuracy(scores, top1_labels_100, k)
    assert res == [1.0]

    # top1 acc, top2 acc
    k = (1, 2)
    top2_labels_0_100 = [3, 1, 1, 1]
    top2_labels_25_75 = [3, 1, 2, 3]
    res = top_k_accuracy(scores, top2_labels_0_100, k)
    assert res == [0, 1.0]
    res = top_k_accuracy(scores, top2_labels_25_75, k)
    assert res == [0.25, 0.75]

    # top1 acc, top3 acc, top5 acc
    k = (1, 3, 5)
    top5_labels_0_0_100 = [1, 0, 3, 2]
    top5_labels_0_50_100 = [1, 3, 4, 0]
    top5_labels_25_75_100 = [2, 3, 0, 2]
    res = top_k_accuracy(scores, top5_labels_0_0_100, k)
    assert res == [0, 0, 1.0]
    res = top_k_accuracy(scores, top5_labels_0_50_100, k)
    assert res == [0, 0.5, 1.0]
    res = top_k_accuracy(scores, top5_labels_25_75_100, k)
    assert res == [0.25, 0.75, 1.0]


def test_mean_class_accuracy():
    scores = [
        np.array([-0.2203, -0.7538, 1.8789, 0.4451, -0.2526]),
        np.array([-0.0413, 0.6366, 1.1155, 0.3484, 0.0395]),
        np.array([0.0365, 0.5158, 1.1067, -0.9276, -0.2124]),
        np.array([0.6232, 0.9912, -0.8562, 0.0148, 1.6413])
    ]

    # test mean class accuracy in [0, 0.25, 1/3, 0.75, 1.0]
    mean_cls_acc_0 = np.int64([1, 4, 0, 2])
    mean_cls_acc_25 = np.int64([2, 0, 4, 3])
    mean_cls_acc_33 = np.int64([2, 2, 2, 3])
    mean_cls_acc_75 = np.int64([4, 2, 2, 4])
    mean_cls_acc_100 = np.int64([2, 2, 2, 4])
    assert mean_class_accuracy(scores, mean_cls_acc_0) == 0
    assert mean_class_accuracy(scores, mean_cls_acc_25) == 0.25
    assert mean_class_accuracy(scores, mean_cls_acc_33) == 1 / 3
    assert mean_class_accuracy(scores, mean_cls_acc_75) == 0.75
    assert mean_class_accuracy(scores, mean_cls_acc_100) == 1.0


def test_mmit_mean_average_precision():
    # One sample
    y_true = [np.array([0, 0, 1, 1])]
    y_scores = [np.array([0.1, 0.4, 0.35, 0.8])]
    map = mmit_mean_average_precision(y_scores, y_true)

    precision = [2.0 / 3.0, 0.5, 1., 1.]
    recall = [1., 0.5, 0.5, 0.]
    target = -np.sum(np.diff(recall) * np.array(precision)[:-1])
    assert target == map


def test_pairwise_temporal_iou():
    target_segments = np.array([])
    candidate_segments = np.array([])
    with pytest.raises(ValueError):
        pairwise_temporal_iou(target_segments, candidate_segments)

    # test temporal iou
    target_segments = np.array([[1, 2], [2, 3]])
    candidate_segments = np.array([[2, 3], [2.5, 3]])
    temporal_iou = pairwise_temporal_iou(candidate_segments, target_segments)
    assert_array_equal(temporal_iou, [[0, 0], [1, 0.5]])

    # test temporal overlap_self
    target_segments = np.array([[1, 2], [2, 3]])
    candidate_segments = np.array([[2, 3], [2.5, 3]])
    temporal_iou, temporal_overlap_self = pairwise_temporal_iou(
        candidate_segments, target_segments, calculate_overlap_self=True)
    assert_array_equal(temporal_overlap_self, [[0, 0], [1, 1]])

    # test temporal overlap_self when candidate_segments is 1d
    target_segments = np.array([[1, 2], [2, 3]])
    candidate_segments = np.array([2.5, 3])
    temporal_iou, temporal_overlap_self = pairwise_temporal_iou(
        candidate_segments, target_segments, calculate_overlap_self=True)
    assert_array_equal(temporal_overlap_self, [0, 1])


def test_average_recall_at_avg_proposals():
    ground_truth1 = {
        'v_test1': np.array([[0, 1], [1, 2]]),
        'v_test2': np.array([[0, 1], [1, 2]])
    }
    ground_truth2 = {'v_test1': np.array([[0, 1]])}
    proposals1 = {
        'v_test1': np.array([[0, 1, 1], [1, 2, 1]]),
        'v_test2': np.array([[0, 1, 1], [1, 2, 1]])
    }
    proposals2 = {
        'v_test1': np.array([[10, 11, 0.6], [11, 12, 0.4]]),
        'v_test2': np.array([[10, 11, 0.6], [11, 12, 0.4]])
    }
    proposals3 = {
        'v_test1': np.array([[i, i + 1, 1 / (i + 1)] for i in range(100)])
    }

    recall, avg_recall, proposals_per_video, auc = (
        average_recall_at_avg_proposals(ground_truth1, proposals1, 4))
    assert_array_equal(recall, [[0.] * 49 + [0.5] * 50 + [1.]] * 10)
    assert_array_equal(avg_recall, [0.] * 49 + [0.5] * 50 + [1.])
    assert_array_almost_equal(
        proposals_per_video, np.arange(0.02, 2.02, 0.02), decimal=10)
    assert auc == 25.5

    recall, avg_recall, proposals_per_video, auc = (
        average_recall_at_avg_proposals(ground_truth1, proposals2, 4))
    assert_array_equal(recall, [[0.] * 100] * 10)
    assert_array_equal(avg_recall, [0.] * 100)
    assert_array_almost_equal(
        proposals_per_video, np.arange(0.02, 2.02, 0.02), decimal=10)
    assert auc == 0

    recall, avg_recall, proposals_per_video, auc = (
        average_recall_at_avg_proposals(ground_truth2, proposals3, 100))
    assert_array_equal(recall, [[1.] * 100] * 10)
    assert_array_equal(avg_recall, ([1.] * 100))
    assert_array_almost_equal(
        proposals_per_video, np.arange(1, 101, 1), decimal=10)
    assert auc == 99.0


def test_get_weighted_score():
    score_a = [
        np.array([-0.2203, -0.7538, 1.8789, 0.4451, -0.2526]),
        np.array([-0.0413, 0.6366, 1.1155, 0.3484, 0.0395]),
        np.array([0.0365, 0.5158, 1.1067, -0.9276, -0.2124]),
        np.array([0.6232, 0.9912, -0.8562, 0.0148, 1.6413])
    ]
    score_b = [
        np.array([-0.0413, 0.6366, 1.1155, 0.3484, 0.0395]),
        np.array([0.0365, 0.5158, 1.1067, -0.9276, -0.2124]),
        np.array([0.6232, 0.9912, -0.8562, 0.0148, 1.6413]),
        np.array([-0.2203, -0.7538, 1.8789, 0.4451, -0.2526])
    ]
    weighted_score = get_weighted_score([score_a], [1])
    assert np.all(np.isclose(np.array(score_a), np.array(weighted_score)))
    coeff_a, coeff_b = 2., 1.
    weighted_score = get_weighted_score([score_a, score_b], [coeff_a, coeff_b])
    ground_truth = [
        x * coeff_a + y * coeff_b for x, y in zip(score_a, score_b)
    ]
    assert np.all(np.isclose(np.array(ground_truth), np.array(weighted_score)))


def test_mean_average_precision():

    def content_for_unittest(scores, labels, result):
        gt = mean_average_precision(scores, labels)
        assert gt == result

    scores = [
        np.array([0.1, 0.2, 0.3, 0.4]),
        np.array([0.2, 0.3, 0.4, 0.1]),
        np.array([0.3, 0.4, 0.1, 0.2]),
        np.array([0.4, 0.1, 0.2, 0.3])
    ]

    label1 = np.array([[1, 1, 0, 0], [1, 0, 1, 1], [1, 0, 1, 0], [1, 1, 0, 1]])
    result1 = 2 / 3
    label2 = np.array([[0, 1, 0, 1], [0, 1, 1, 0], [1, 0, 1, 0], [0, 0, 1, 1]])
    result2 = np.mean([0.5, 0.5833333333333333, 0.8055555555555556, 1.0])

    content_for_unittest(scores, label1, result1)
    content_for_unittest(scores, label2, result2)
