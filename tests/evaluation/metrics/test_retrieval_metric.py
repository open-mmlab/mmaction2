# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import numpy as np
import pytest
import torch

from mmaction.evaluation.metrics import RetrievalMetric, RetrievalRecall
from mmaction.registry import METRICS
from mmaction.structures import ActionDataSample


def generate_data(num_samples=5, feat_dim=10, random_label=False):
    data_batch = []
    data_samples = []
    for i in range(num_samples):
        if random_label:
            video_feature = torch.randn(feat_dim)
            text_feature = torch.randn(feat_dim)
        else:
            video_feature = torch.randn(feat_dim)
            text_feature = video_feature.clone()

        data_sample = dict(
            features=dict(
                video_feature=video_feature, text_feature=text_feature))
        data_samples.append(data_sample)
    return data_batch, data_samples


def test_acc_metric():
    with pytest.raises(ValueError):
        RetrievalMetric(metric_list='R100')

    num_samples = 20
    metric = RetrievalMetric()
    data_batch, predictions = generate_data(
        num_samples=num_samples, random_label=True)
    metric.process(data_batch, predictions)
    eval_results = metric.compute_metrics(metric.results)
    assert 0.0 <= eval_results['R1'] <= eval_results['R5'] <= eval_results[
        'R10'] <= 100.0
    assert 0.0 <= eval_results['MdR'] <= num_samples
    assert 0.0 <= eval_results['MnR'] <= num_samples

    metric.results.clear()

    data_batch, predictions = generate_data(
        num_samples=num_samples, random_label=False)
    metric.process(data_batch, predictions)
    eval_results = metric.compute_metrics(metric.results)
    assert eval_results['R1'] == eval_results['R5'] == eval_results[
        'R10'] == 100.0
    assert eval_results['MdR'] == eval_results['MnR'] == 1.0


class TestRetrievalRecall(TestCase):

    def test_evaluate(self):
        """Test using the metric in the same way as Evalutor."""
        pred = [
            ActionDataSample().set_pred_score(i).set_gt_label(k).to_dict()
            for i, k in zip([
                torch.tensor([0.7, 0.0, 0.3]),
                torch.tensor([0.5, 0.2, 0.3]),
                torch.tensor([0.4, 0.5, 0.1]),
                torch.tensor([0.0, 0.0, 1.0]),
                torch.tensor([0.0, 0.0, 1.0]),
                torch.tensor([0.0, 0.0, 1.0]),
            ], [[0], [0], [1], [2], [2], [0]])
        ]

        # Test with score (use score instead of label if score exists)
        metric = METRICS.build(dict(type='RetrievalRecall', topk=1))
        metric.process(None, pred)
        recall = metric.evaluate(6)
        self.assertIsInstance(recall, dict)
        self.assertAlmostEqual(
            recall['retrieval/Recall@1'], 5 / 6 * 100, places=4)

        # Test with invalid topk
        with self.assertRaisesRegex(RuntimeError, 'selected index k'):
            metric = METRICS.build(dict(type='RetrievalRecall', topk=10))
            metric.process(None, pred)
            metric.evaluate(6)

        with self.assertRaisesRegex(ValueError, '`topk` must be a'):
            METRICS.build(dict(type='RetrievalRecall', topk=-1))

        # Test initialization
        metric = METRICS.build(dict(type='RetrievalRecall', topk=5))
        self.assertEqual(metric.topk, (5, ))

        # Test initialization
        metric = METRICS.build(dict(type='RetrievalRecall', topk=(1, 2, 5)))
        self.assertEqual(metric.topk, (1, 2, 5))

    def test_calculate(self):
        """Test using the metric from static method."""

        # seq of indices format
        y_true = [[0, 2, 5, 8, 9], [1, 4, 6]]
        y_pred = [np.arange(10)] * 2

        # test with average is 'macro'
        recall_score = RetrievalRecall.calculate(
            y_pred, y_true, topk=1, pred_indices=True, target_indices=True)
        expect_recall = 50.
        self.assertEqual(recall_score[0].item(), expect_recall)

        # test with tensor input
        y_true = torch.Tensor([[1, 0, 1, 0, 0, 1, 0, 0, 1, 1],
                               [0, 1, 0, 0, 1, 0, 1, 0, 0, 0]])
        y_pred = np.array([np.linspace(0.95, 0.05, 10)] * 2)
        recall_score = RetrievalRecall.calculate(y_pred, y_true, topk=1)
        expect_recall = 50.
        self.assertEqual(recall_score[0].item(), expect_recall)

        # test with topk is 5
        y_pred = np.array([np.linspace(0.95, 0.05, 10)] * 2)
        recall_score = RetrievalRecall.calculate(y_pred, y_true, topk=2)
        expect_recall = 100.
        self.assertEqual(recall_score[0].item(), expect_recall)

        # test with topk is (1, 5)
        y_pred = np.array([np.linspace(0.95, 0.05, 10)] * 2)
        recall_score = RetrievalRecall.calculate(y_pred, y_true, topk=(1, 5))
        expect_recalls = [50., 100.]
        self.assertEqual(len(recall_score), len(expect_recalls))
        for i in range(len(expect_recalls)):
            self.assertEqual(recall_score[i].item(), expect_recalls[i])

        # Test with invalid pred
        y_pred = dict()
        y_true = [[0, 2, 5, 8, 9], [1, 4, 6]]
        with self.assertRaisesRegex(AssertionError, '`pred` must be Seq'):
            RetrievalRecall.calculate(y_pred, y_true, True, True)

        # Test with invalid target
        y_true = dict()
        y_pred = [np.arange(10)] * 2
        with self.assertRaisesRegex(AssertionError, '`target` must be Seq'):
            RetrievalRecall.calculate(
                y_pred, y_true, topk=1, pred_indices=True, target_indices=True)

        # Test with different length `pred` with `target`
        y_true = [[0, 2, 5, 8, 9], [1, 4, 6]]
        y_pred = [np.arange(10)] * 3
        with self.assertRaisesRegex(AssertionError, 'Length of `pred`'):
            RetrievalRecall.calculate(
                y_pred, y_true, topk=1, pred_indices=True, target_indices=True)

        # Test with invalid pred
        y_true = [[0, 2, 5, 8, 9], dict()]
        y_pred = [np.arange(10)] * 2
        with self.assertRaisesRegex(AssertionError, '`target` should be'):
            RetrievalRecall.calculate(
                y_pred, y_true, topk=1, pred_indices=True, target_indices=True)

        # Test with invalid target
        y_true = [[0, 2, 5, 8, 9], [1, 4, 6]]
        y_pred = [np.arange(10), dict()]
        with self.assertRaisesRegex(AssertionError, '`pred` should be'):
            RetrievalRecall.calculate(
                y_pred, y_true, topk=1, pred_indices=True, target_indices=True)
