# Copyright (c) OpenMMLab. All rights reserved.
"""import os.path as osp.

import torch

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
        osp.join(osp.dirname(__file__), '../../../data/eval_detection'))
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
"""
