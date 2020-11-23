import os.path as osp
from abc import abstractproperty

import numpy as np
import torch

from mmaction.core.bbox import (bbox2result, bbox2roi, bbox_overlaps,
                                bbox_target, build_assigner, build_sampler)
from mmaction.datasets import AVADataset


def test_assigner_sampler():
    data_prefix = osp.join(osp.dirname(__file__), 'data/test_eval_detection')
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
        type='MaxIoUAssigner',
        pos_iou_thr=0.75,
        neg_iou_thr=0.75,
        min_pos_iou=0.75)
    assigner = build_assigner(assigner)
    proposal = torch.tensor(dataset[0]['proposals'])
    gt_bboxes = torch.tensor(dataset[0]['entity_boxes'])
    gt_labels = torch.tensor(dataset[0]['labels'])
    assign_result = assigner.assign(proposal, gt_bboxes, gt_labels)
    assert assign_result.num_gts == 4
    assert torch.all(
        assign_result.gt_inds == torch.tensor([0, 1, 2, 3, 0, 0, 0, 1, 0, 0]))
    assert torch.all(
        torch.isclose(
            assign_result.max_overlaps,
            torch.tensor([
                0.69276601, 0.77789903, 0.78705092, 0.82503866, 0.71533428,
                0.72312719, 0.72436035, 0.78155030, 0.69307099, 0.68858184
            ],
                         dtype=torch.float64)))
    assert torch.all(
        torch.isclose(
            assign_result.labels,
            torch.tensor([[0., 0., 0., 0.], [0., 0., 0., 1.], [0., 1., 0., 0.],
                          [0., 1., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.],
                          [0., 0., 0., 0.], [0., 0., 0., 1.], [0., 0., 0., 0.],
                          [0., 0., 0., 0.]])))
    sampler = dict(type='RandomSampler', num=32, pos_fraction=1)
    sampler = build_sampler(sampler)
    sampling_result = sampler.sample(assign_result, proposal, gt_bboxes,
                                     gt_labels)

    assert sampling_result is not None


def test_bbox_overlaps():
    dtype = torch.float
    b1 = torch.tensor([[1.0, 1.0, 3.0, 4.0], [2.0, 2.0, 3.0, 4.0],
                       [7.0, 7.0, 8.0, 8.0]]).cuda().type(dtype)
    b2 = torch.tensor([[0.0, 2.0, 2.0, 5.0], [2.0, 1.0, 3.0,
                                              3.0]]).cuda().type(dtype)
    should_output = np.array([[0.33333334, 0.5], [0.2, 0.5], [0.0, 0.0]])
    out = bbox_overlaps(b1, b2)
    assert np.allclose(out.cpu().numpy(), should_output, 1e-2)

    b1 = torch.tensor([[1.0, 1.0, 3.0, 4.0], [2.0, 2.0, 3.0,
                                              4.0]]).cuda().type(dtype)
    b2 = torch.tensor([[0.0, 2.0, 2.0, 5.0], [2.0, 1.0, 3.0,
                                              3.0]]).cuda().type(dtype)
    should_output = np.array([0.33333334, 0.5])
    out = bbox_overlaps(b1, b2, aligned=True)
    assert np.allclose(out.cpu().numpy(), should_output, 1e-2)

    b1 = torch.tensor([[0.0, 0.0, 3.0, 3.0]]).cuda().type(dtype)
    b1 = torch.tensor([[0.0, 0.0, 3.0, 3.0]]).cuda().type(dtype)
    b2 = torch.tensor([[4.0, 0.0, 5.0, 3.0], [3.0, 0.0, 4.0, 3.0],
                       [2.0, 0.0, 3.0, 3.0], [1.0, 0.0, 2.0,
                                              3.0]]).cuda().type(dtype)
    should_output = np.array([0, 0.2, 0.5, 0.5])
    out = bbox_overlaps(
        b1,
        b2,
    )
    assert np.allclose(out.cpu().numpy(), should_output, 1e-2)


def test_bbox2roi():
    bbox_list = [
        torch.tensor([[0, 0, 1, 1]]),
        torch.tensor([[0.1, 0.1, 0.9, 0.9]]),
        torch.tensor([[0.2, 0.2, 0.8, 0.8], [0.3, 0.3, 0.7, 0.7]])
    ]
    rois = bbox2roi(bbox_list)
    assert torch.all(
        torch.isclose(
            rois,
            torch.tensor([[0.00, 0.00, 0.00, 1.00, 1.00],
                          [1.00, 0.10, 0.00, 0.90, 0.90],
                          [2.00, 0.20, 0.00, 0.80, 0.80],
                          [2.00, 0.30, 0.00, 0.70, 0.70]])))


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
            torch.tensor([[0., 0., 0., 0.], [0., 0., 0., 1.], [0., 1., 0., 0.],
                          [0., 1., 0., 0.], [0., 0., 0., 0.], [0., 0., 0.,
                                                               0.]])))
    assert torch.all(
        torch.isclose(label_weights, torch.tensor([0.8] * 4 + [1.0] * 2)))
