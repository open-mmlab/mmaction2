# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import platform
from unittest import TestCase

import numpy as np
import pytest
import torch
from mmengine import load
from numpy.testing import assert_array_almost_equal

from mmaction.evaluation import AccMetric, ConfusionMatrix, MultiSportsMetric
from mmaction.evaluation.functional import ava_eval
from mmaction.registry import METRICS
from mmaction.structures import ActionDataSample


def generate_data(num_classes=5, random_label=False, multi_label=False):
    data_batch = []
    data_samples = []
    for i in range(num_classes * 10):
        scores = torch.randn(num_classes)
        if multi_label:
            label = torch.ones_like(scores)
        elif random_label:
            label = torch.randint(num_classes, size=[1])
        else:
            label = torch.LongTensor([scores.argmax().item()])
        data_sample = dict(pred_score=scores, gt_label=label)
        data_samples.append(data_sample)
    return data_batch, data_samples


def test_acc_metric():
    num_classes = 32
    metric = AccMetric(metric_list=('top_k_accuracy', 'mean_class_accuracy'))
    data_batch, predictions = generate_data(
        num_classes=num_classes, random_label=True)
    metric.process(data_batch, predictions)
    eval_results = metric.compute_metrics(metric.results)
    assert 0.0 <= eval_results['top1'] <= eval_results['top5'] <= 1.0
    assert 0.0 <= eval_results['mean1'] <= 1.0
    metric.results.clear()

    data_batch, predictions = generate_data(
        num_classes=num_classes, random_label=False)
    metric.process(data_batch, predictions)
    eval_results = metric.compute_metrics(metric.results)
    assert eval_results['top1'] == eval_results['top5'] == 1.0
    assert eval_results['mean1'] == 1.0

    metric = AccMetric(
        metric_list=('mean_average_precision', 'mmit_mean_average_precision'))
    data_batch, predictions = generate_data(
        num_classes=num_classes, multi_label=True)
    metric.process(data_batch, predictions)
    eval_results = metric.compute_metrics(metric.results)
    assert eval_results['mean_average_precision'] == 1.0
    assert eval_results['mmit_mean_average_precision'] == 1.0


@pytest.mark.skipif(platform.system() == 'Windows', reason='Multiprocess Fail')
def test_ava_detection():
    data_prefix = osp.normpath(
        osp.join(osp.dirname(__file__), '../../data/eval_detection'))

    gt_path = osp.join(data_prefix, 'gt.csv')
    result_path = osp.join(data_prefix, 'pred.csv')
    label_map = osp.join(data_prefix, 'action_list.txt')

    # eval bbox
    detection = ava_eval(result_path, 'mAP', label_map, gt_path, None)
    assert_array_almost_equal(detection['overall'], 0.09385522)


def test_multisport_detection():
    data_prefix = osp.normpath(
        osp.join(osp.dirname(__file__), '../../data/eval_multisports'))

    gt_path = osp.join(data_prefix, 'gt.pkl')
    result_path = osp.join(data_prefix, 'data_samples.pkl')

    result_datasamples = load(result_path)
    metric = MultiSportsMetric(gt_path)
    metric.process(None, result_datasamples)
    eval_result = metric.compute_metrics(metric.results)
    assert eval_result['frameAP'] == 83.6506
    assert eval_result['v_map@0.2'] == 37.5
    assert eval_result['v_map@0.5'] == 37.5
    assert eval_result['v_map_0.10:0.90'] == 29.1667


class TestConfusionMatrix(TestCase):

    def test_evaluate(self):
        """Test using the metric in the same way as Evalutor."""
        pred = [
            ActionDataSample().set_pred_score(i).set_pred_label(
                j).set_gt_label(k).to_dict() for i, j, k in zip([
                    torch.tensor([0.7, 0.0, 0.3]),
                    torch.tensor([0.5, 0.2, 0.3]),
                    torch.tensor([0.4, 0.5, 0.1]),
                    torch.tensor([0.0, 0.0, 1.0]),
                    torch.tensor([0.0, 0.0, 1.0]),
                    torch.tensor([0.0, 0.0, 1.0]),
                ], [0, 0, 1, 2, 2, 2], [0, 0, 1, 2, 1, 0])
        ]

        # Test with score (use score instead of label if score exists)
        metric = METRICS.build(dict(type='ConfusionMatrix'))
        metric.process(None, pred)
        res = metric.evaluate(6)
        self.assertIsInstance(res, dict)
        self.assertTensorEqual(
            res['confusion_matrix/result'],
            torch.tensor([
                [2, 0, 1],
                [0, 1, 1],
                [0, 0, 1],
            ]))

        # Test with label
        for sample in pred:
            del sample['pred_score']
        metric = METRICS.build(dict(type='ConfusionMatrix'))
        metric.process(None, pred)
        with self.assertRaisesRegex(AssertionError,
                                    'Please specify the `num_classes`'):
            metric.evaluate(6)

        metric = METRICS.build(dict(type='ConfusionMatrix', num_classes=3))
        metric.process(None, pred)
        self.assertIsInstance(res, dict)
        self.assertTensorEqual(
            res['confusion_matrix/result'],
            torch.tensor([
                [2, 0, 1],
                [0, 1, 1],
                [0, 0, 1],
            ]))

    def test_calculate(self):
        y_true = np.array([0, 0, 1, 2, 1, 0])
        y_label = torch.tensor([0, 0, 1, 2, 2, 2])
        y_score = [
            [0.7, 0.0, 0.3],
            [0.5, 0.2, 0.3],
            [0.4, 0.5, 0.1],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]

        # Test with score
        cm = ConfusionMatrix.calculate(y_score, y_true)
        self.assertIsInstance(cm, torch.Tensor)
        self.assertTensorEqual(
            cm, torch.tensor([
                [2, 0, 1],
                [0, 1, 1],
                [0, 0, 1],
            ]))

        # Test with label
        with self.assertRaisesRegex(AssertionError,
                                    'Please specify the `num_classes`'):
            ConfusionMatrix.calculate(y_label, y_true)

        cm = ConfusionMatrix.calculate(y_label, y_true, num_classes=3)
        self.assertIsInstance(cm, torch.Tensor)
        self.assertTensorEqual(
            cm, torch.tensor([
                [2, 0, 1],
                [0, 1, 1],
                [0, 0, 1],
            ]))

        # Test with invalid inputs
        with self.assertRaisesRegex(TypeError, "<class 'str'> is not"):
            ConfusionMatrix.calculate(y_label, 'hi')

    def test_plot(self):
        import matplotlib.pyplot as plt

        cm = torch.tensor([[2, 0, 1], [0, 1, 1], [0, 0, 1]])
        fig = ConfusionMatrix.plot(cm, include_values=True, show=False)

        self.assertIsInstance(fig, plt.Figure)

    def assertTensorEqual(self,
                          tensor: torch.Tensor,
                          value: float,
                          msg=None,
                          **kwarg):
        tensor = tensor.to(torch.float32)
        value = torch.tensor(value).float()
        try:
            torch.testing.assert_allclose(tensor, value, **kwarg)
        except AssertionError as e:
            self.fail(self._formatMessage(msg, str(e)))
