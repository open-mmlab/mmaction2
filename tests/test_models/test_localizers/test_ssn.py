import copy

import mmcv
import pytest
import torch

from mmaction.models import build_localizer


def test_ssn_train():
    train_cfg = mmcv.ConfigDict(
        dict(
            ssn=dict(
                assigner=dict(
                    positive_iou_threshold=0.7,
                    background_iou_threshold=0.01,
                    incomplete_iou_threshold=0.3,
                    background_coverage_threshold=0.02,
                    incomplete_overlap_threshold=0.01),
                sampler=dict(
                    num_per_video=8,
                    positive_ratio=1,
                    background_ratio=1,
                    incomplete_ratio=6,
                    add_gt_as_proposals=True),
                loss_weight=dict(comp_loss_weight=0.1, reg_loss_weight=0.1),
                debug=False)))
    base_model_cfg = dict(
        type='SSN',
        backbone=dict(
            type='ResNet', pretrained=None, depth=18, norm_eval=True),
        spatial_type='avg',
        dropout_ratio=0.8,
        loss_cls=dict(type='SSNLoss'),
        cls_head=dict(
            type='SSNHead',
            dropout_ratio=0.,
            in_channels=512,
            num_classes=20,
            consensus=dict(
                type='STPPTrain',
                stpp_stage=(1, 1, 1),
                num_segments_list=(2, 5, 2)),
            use_regression=True),
        train_cfg=train_cfg)
    dropout_cfg = copy.deepcopy(base_model_cfg)
    dropout_cfg['dropout_ratio'] = 0
    dropout_cfg['cls_head']['dropout_ratio'] = 0.5
    non_regression_cfg = copy.deepcopy(base_model_cfg)
    non_regression_cfg['cls_head']['use_regression'] = False

    imgs = torch.rand(1, 8, 9, 3, 224, 224)
    proposal_scale_factor = torch.Tensor([[[1.0345, 1.0345], [1.0028, 0.0028],
                                           [1.0013, 1.0013], [1.0008, 1.0008],
                                           [0.3357, 1.0006], [1.0006, 1.0006],
                                           [0.0818, 1.0005], [1.0030,
                                                              1.0030]]])
    proposal_type = torch.Tensor([[0, 1, 1, 1, 1, 1, 1, 2]])
    proposal_labels = torch.LongTensor([[8, 8, 8, 8, 8, 8, 8, 0]])
    reg_targets = torch.Tensor([[[0.2929, 0.2694], [0.0000, 0.0000],
                                 [0.0000, 0.0000], [0.0000, 0.0000],
                                 [0.0000, 0.0000], [0.0000, 0.0000],
                                 [0.0000, 0.0000], [0.0000, 0.0000]]])

    localizer_ssn = build_localizer(base_model_cfg)
    localizer_ssn_dropout = build_localizer(dropout_cfg)
    localizer_ssn_non_regression = build_localizer(non_regression_cfg)

    if torch.cuda.is_available():
        localizer_ssn = localizer_ssn.cuda()
        localizer_ssn_dropout = localizer_ssn_dropout.cuda()
        localizer_ssn_non_regression = localizer_ssn_non_regression.cuda()
        imgs = imgs.cuda()
        proposal_scale_factor = proposal_scale_factor.cuda()
        proposal_type = proposal_type.cuda()
        proposal_labels = proposal_labels.cuda()
        reg_targets = reg_targets.cuda()

    # Train normal case
    losses = localizer_ssn(
        imgs,
        proposal_scale_factor=proposal_scale_factor,
        proposal_type=proposal_type,
        proposal_labels=proposal_labels,
        reg_targets=reg_targets)
    assert isinstance(losses, dict)

    # Train SSN without dropout in model, with dropout in head
    losses = localizer_ssn_dropout(
        imgs,
        proposal_scale_factor=proposal_scale_factor,
        proposal_type=proposal_type,
        proposal_labels=proposal_labels,
        reg_targets=reg_targets)
    assert isinstance(losses, dict)

    # Train SSN model without regression
    losses = localizer_ssn_non_regression(
        imgs,
        proposal_scale_factor=proposal_scale_factor,
        proposal_type=proposal_type,
        proposal_labels=proposal_labels,
        reg_targets=reg_targets)
    assert isinstance(losses, dict)


def test_ssn_test():
    test_cfg = mmcv.ConfigDict(
        dict(
            ssn=dict(
                sampler=dict(test_interval=6, batch_size=16),
                evaluater=dict(
                    top_k=2000,
                    nms=0.2,
                    softmax_before_filter=True,
                    cls_score_dict=None,
                    cls_top_k=2))))
    base_model_cfg = dict(
        type='SSN',
        backbone=dict(
            type='ResNet', pretrained=None, depth=18, norm_eval=True),
        spatial_type='avg',
        dropout_ratio=0.8,
        cls_head=dict(
            type='SSNHead',
            dropout_ratio=0.,
            in_channels=512,
            num_classes=20,
            consensus=dict(type='STPPTest', stpp_stage=(1, 1, 1)),
            use_regression=True),
        test_cfg=test_cfg)
    maxpool_model_cfg = copy.deepcopy(base_model_cfg)
    maxpool_model_cfg['spatial_type'] = 'max'
    non_regression_cfg = copy.deepcopy(base_model_cfg)
    non_regression_cfg['cls_head']['use_regression'] = False
    non_regression_cfg['cls_head']['consensus']['use_regression'] = False
    tuple_stage_cfg = copy.deepcopy(base_model_cfg)
    tuple_stage_cfg['cls_head']['consensus']['stpp_stage'] = (1, (1, 2), 1)
    str_stage_cfg = copy.deepcopy(base_model_cfg)
    str_stage_cfg['cls_head']['consensus']['stpp_stage'] = ('error', )

    imgs = torch.rand(1, 8, 3, 224, 224)
    relative_proposal_list = torch.Tensor([[[0.2500, 0.6250], [0.3750,
                                                               0.7500]]])
    scale_factor_list = torch.Tensor([[[1.0000, 1.0000], [1.0000, 0.2661]]])
    proposal_tick_list = torch.LongTensor([[[1, 2, 5, 7], [20, 30, 60, 80]]])
    reg_norm_consts = torch.Tensor([[[-0.0603, 0.0325], [0.0752, 0.1596]]])

    localizer_ssn = build_localizer(base_model_cfg)
    localizer_ssn_maxpool = build_localizer(maxpool_model_cfg)
    localizer_ssn_non_regression = build_localizer(non_regression_cfg)
    localizer_ssn_tuple_stage_cfg = build_localizer(tuple_stage_cfg)
    with pytest.raises(ValueError):
        build_localizer(str_stage_cfg)

    if torch.cuda.is_available():
        localizer_ssn = localizer_ssn.cuda()
        localizer_ssn_maxpool = localizer_ssn_maxpool.cuda()
        localizer_ssn_non_regression = localizer_ssn_non_regression.cuda()
        localizer_ssn_tuple_stage_cfg = localizer_ssn_tuple_stage_cfg.cuda()
        imgs = imgs.cuda()
        relative_proposal_list = relative_proposal_list.cuda()
        scale_factor_list = scale_factor_list.cuda()
        proposal_tick_list = proposal_tick_list.cuda()
        reg_norm_consts = reg_norm_consts.cuda()

    with torch.no_grad():
        # Test normal case
        localizer_ssn(
            imgs,
            relative_proposal_list=relative_proposal_list,
            scale_factor_list=scale_factor_list,
            proposal_tick_list=proposal_tick_list,
            reg_norm_consts=reg_norm_consts,
            return_loss=False)

        # Test SSN model with max spatial pooling
        localizer_ssn_maxpool(
            imgs,
            relative_proposal_list=relative_proposal_list,
            scale_factor_list=scale_factor_list,
            proposal_tick_list=proposal_tick_list,
            reg_norm_consts=reg_norm_consts,
            return_loss=False)

        # Test SSN model without regression
        localizer_ssn_non_regression(
            imgs,
            relative_proposal_list=relative_proposal_list,
            scale_factor_list=scale_factor_list,
            proposal_tick_list=proposal_tick_list,
            reg_norm_consts=reg_norm_consts,
            return_loss=False)

        # Test SSN model with tuple stage cfg.
        localizer_ssn_tuple_stage_cfg(
            imgs,
            relative_proposal_list=relative_proposal_list,
            scale_factor_list=scale_factor_list,
            proposal_tick_list=proposal_tick_list,
            reg_norm_consts=reg_norm_consts,
            return_loss=False)
