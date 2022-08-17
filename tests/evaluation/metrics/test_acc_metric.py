# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmaction.evaluation import AccMetric


def generate_data(num_classes=5, random_label=False):
    data_batch = []
    predictions = []
    for i in range(num_classes * 10):
        logit = torch.randn(num_classes)
        pred = dict(pred_scores=dict(item=logit))
        predictions.append(pred)
        if random_label:
            label = torch.randint(num_classes, size=[])
        else:
            label = torch.tensor(logit.argmax().item())
        data = dict(data_sample=dict(gt_labels=dict(item=label)))
        data_batch.append(data)
    return data_batch, predictions


def test_accmetric():
    num_classes = 32
    metric = AccMetric(
        metric_list=('top_k_accuracy', 'mean_class_accuracy',
                     'mmit_mean_average_precision', 'mean_average_precision'),
        num_classes=num_classes)
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
    assert eval_results['mmit_mean_average_precision'] == 1.0
    return
