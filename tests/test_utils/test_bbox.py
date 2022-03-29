# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from abc import abstractproperty

import numpy as np
import torch

from mmaction.core.bbox import bbox2result, bbox_target
from mmaction.datasets import AVADataset


def test_assigner_sampler():
    try:
        from mmdet.core.bbox import build_assigner, build_sampler
    except (ImportError, ModuleNotFoundError):
        raise ImportError(
            'Failed to import `build_assigner` and `build_sampler` '
            'from `mmdet.core.bbox`. The two APIs are required for '
            'the testing in `test_bbox.py`! ')
    data_prefix = osp.normpath(
        osp.join(osp.dirname(__file__), '../data/eval_detection'))
    ann_file = osp.join(data_prefix, 'gt.csv')
    label_file = osp.join(data_prefix, 'action_list.txt')
    proposal_file = osp.join(data_prefix, 'proposal.pkl')
    dataset = AVADataset(
        ann_file=ann_file,
        exclude_file=None,
        pipeline=[],
        label_file=label_file,
        proposal_file=proposal_file,
        num_classes=4)

    assigner = dict(
        type='MaxIoUAssignerAVA',
        pos_iou_thr=0.5,
        neg_iou_thr=0.5,
        min_pos_iou=0.5)
    assigner = build_assigner(assigner)
    proposal = torch.tensor(dataset[0]['proposals'])

    gt_bboxes = torch.tensor(dataset[0]['gt_bboxes'])
    gt_labels = torch.tensor(dataset[0]['gt_labels'])
    assign_result = assigner.assign(
        bboxes=proposal,
        gt_bboxes=gt_bboxes,
        gt_bboxes_ignore=None,
        gt_labels=gt_labels)
    assert assign_result.num_gts == 4
    assert torch.all(
        assign_result.gt_inds == torch.tensor([0, 0, 3, 3, 0, 0, 0, 1, 0, 0]))
    assert torch.all(
        torch.isclose(
            assign_result.max_overlaps,
            torch.tensor([
                0.40386841, 0.47127257, 0.53544776, 0.58797631, 0.29281288,
                0.40979504, 0.45902917, 0.50093938, 0.21560125, 0.32948171
            ],
                         dtype=torch.float64)))
    assert torch.all(
        torch.isclose(
            assign_result.labels,
            torch.tensor([[0., 0., 0., 0.], [0., 0., 0., 0.], [0., 1., 0., 0.],
                          [0., 1., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.],
                          [0., 0., 0., 0.], [0., 0., 0., 1.], [0., 0., 0., 0.],
                          [0., 0., 0., 0.]])))
    sampler = dict(type='RandomSampler', num=32, pos_fraction=1)
    sampler = build_sampler(sampler)
    sampling_result = sampler.sample(assign_result, proposal, gt_bboxes,
                                     gt_labels)
    assert (sampling_result.pos_inds.shape[0] ==
            sampling_result.pos_bboxes.shape[0])
    assert (sampling_result.neg_inds.shape[0] ==
            sampling_result.neg_bboxes.shape[0])
    return sampling_result


def test_bbox2result():
    bboxes = torch.tensor([[0.072, 0.47, 0.84, 0.898],
                           [0.23, 0.215, 0.781, 0.534],
                           [0.195, 0.128, 0.643, 0.944],
                           [0.236, 0.189, 0.689, 0.74],
                           [0.375, 0.371, 0.726, 0.804],
                           [0.024, 0.398, 0.776, 0.719]])
    labels = torch.tensor([[-1.650, 0.515, 0.798, 1.240],
                           [1.368, -1.128, 0.037, -1.087],
                           [0.481, -1.303, 0.501, -0.463],
                           [-0.356, 0.126, -0.840, 0.438],
                           [0.079, 1.269, -0.263, -0.538],
                           [-0.853, 0.391, 0.103, 0.398]])
    num_classes = 4
    #  Test for multi-label
    result = bbox2result(bboxes, labels, num_classes)
    assert np.all(
        np.isclose(
            result[0],
            np.array([[0.072, 0.47, 0.84, 0.898, 0.515],
                      [0.236, 0.189, 0.689, 0.74, 0.126],
                      [0.375, 0.371, 0.726, 0.804, 1.269],
                      [0.024, 0.398, 0.776, 0.719, 0.391]])))
    assert np.all(
        np.isclose(
            result[1],
            np.array([[0.072, 0.47, 0.84, 0.898, 0.798],
                      [0.23, 0.215, 0.781, 0.534, 0.037],
                      [0.195, 0.128, 0.643, 0.944, 0.501],
                      [0.024, 0.398, 0.776, 0.719, 0.103]])))
    assert np.all(
        np.isclose(
            result[2],
            np.array([[0.072, 0.47, 0.84, 0.898, 1.24],
                      [0.236, 0.189, 0.689, 0.74, 0.438],
                      [0.024, 0.398, 0.776, 0.719, 0.398]])))

    # Test for single-label
    result = bbox2result(bboxes, labels, num_classes, -1.0)
    assert np.all(
        np.isclose(result[0], np.array([[0.375, 0.371, 0.726, 0.804, 1.269]])))
    assert np.all(
        np.isclose(
            result[1],
            np.array([[0.23, 0.215, 0.781, 0.534, 0.037],
                      [0.195, 0.128, 0.643, 0.944, 0.501]])))
    assert np.all(
        np.isclose(
            result[2],
            np.array([[0.072, 0.47, 0.84, 0.898, 1.240],
                      [0.236, 0.189, 0.689, 0.74, 0.438],
                      [0.024, 0.398, 0.776, 0.719, 0.398]])))


def test_bbox_target():
    pos_bboxes = torch.tensor([[0.072, 0.47, 0.84, 0.898],
                               [0.23, 0.215, 0.781, 0.534],
                               [0.195, 0.128, 0.643, 0.944],
                               [0.236, 0.189, 0.689, 0.74]])
    neg_bboxes = torch.tensor([[0.375, 0.371, 0.726, 0.804],
                               [0.024, 0.398, 0.776, 0.719]])
    pos_gt_labels = torch.tensor([[0., 0., 1., 0.], [0., 0., 0., 1.],
                                  [0., 1., 0., 0.], [0., 1., 0., 0.]])
    cfg = abstractproperty()
    cfg.pos_weight = 0.8
    labels, label_weights = bbox_target([pos_bboxes], [neg_bboxes],
                                        [pos_gt_labels], cfg)
    assert torch.all(
        torch.isclose(
            labels,
            torch.tensor([[0., 0., 1., 0.], [0., 0., 0., 1.], [0., 1., 0., 0.],
                          [0., 1., 0., 0.], [0., 0., 0., 0.], [0., 0., 0.,
                                                               0.]])))
    assert torch.all(
        torch.isclose(label_weights, torch.tensor([0.8] * 4 + [1.0] * 2)))
