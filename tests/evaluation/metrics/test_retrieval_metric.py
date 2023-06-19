# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmaction.evaluation.metrics import RetrievalMetric


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
