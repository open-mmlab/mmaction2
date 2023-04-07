# Copyright (c) OpenMMLab. All rights reserved.
import random

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from mmaction.evaluation.functional import (average_recall_at_avg_proposals,
                                            confusion_matrix,
                                            get_weighted_score,
                                            pairwise_temporal_iou,
                                            top_k_classes)


def test_top_k_accurate_classes():
    scores = [
        np.array([0.1, 0.2, 0.3, 0.4]),  # 3
        np.array([0.2, 0.3, 0.4, 0.1]),  # 2
        np.array([0.3, 0.4, 0.1, 0.2]),  # 1
        np.array([0.4, 0.1, 0.2, 0.3]),  # 0
        np.array([0.25, 0.1, 0.3, 0.35]),  # 3
        np.array([0.2, 0.15, 0.3, 0.35]),  # 3
    ]
    label = np.array([3, 2, 2, 1, 3, 3], dtype=np.int64)

    with pytest.raises(AssertionError):
        top_k_classes(scores, label, 1, mode='wrong')

    results_top1 = top_k_classes(scores, label, 1)
    results_top3 = top_k_classes(scores, label, 3)
    assert len(results_top1) == 1
    assert len(results_top3) == 3
    assert results_top3[0] == results_top1[0]
    assert results_top1 == [(3, 1.)]
    assert results_top3 == [(3, 1.), (2, 0.5), (1, 0.0)]

    label = np.array([3, 2, 1, 1, 3, 0], dtype=np.int64)
    results_top1 = top_k_classes(scores, label, 1, mode='inaccurate')
    results_top3 = top_k_classes(scores, label, 3, mode='inaccurate')
    assert len(results_top1) == 1
    assert len(results_top3) == 3
    assert results_top3[0] == results_top1[0]
    assert results_top1 == [(0, 0.)]
    assert results_top3 == [(0, 0.0), (1, 0.5), (2, 1.0)]


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
        confusion_mat = np.array(confusion_mat, dtype=np.float64)
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
