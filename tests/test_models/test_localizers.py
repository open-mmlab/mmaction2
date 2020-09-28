import copy

import mmcv
import numpy as np
import pytest
import torch

from mmaction.models import build_localizer
from mmaction.models.localizers.utils import post_processing


def test_tem():
    model_cfg = dict(
        type='TEM',
        temporal_dim=100,
        boundary_ratio=0.1,
        tem_feat_dim=400,
        tem_hidden_dim=512,
        tem_match_threshold=0.5)

    localizer_tem = build_localizer(model_cfg)
    raw_feature = torch.rand(8, 400, 100)
    gt_bbox = torch.Tensor([[[1.0, 3.0], [3.0, 5.0]]] * 8)
    losses = localizer_tem(raw_feature, gt_bbox)
    assert isinstance(losses, dict)

    # Test forward test
    video_meta = [{'video_name': 'v_test'}]
    with torch.no_grad():
        for one_raw_feature in raw_feature:
            one_raw_feature = one_raw_feature.reshape(1, 400, 100)
            localizer_tem(
                one_raw_feature, video_meta=video_meta, return_loss=False)


def test_pem():
    model_cfg = dict(
        type='PEM',
        pem_feat_dim=32,
        pem_hidden_dim=256,
        pem_u_ratio_m=1,
        pem_u_ratio_l=2,
        pem_high_temporal_iou_threshold=0.6,
        pem_low_temporal_iou_threshold=2.2,
        soft_nms_alpha=0.75,
        soft_nms_low_threshold=0.65,
        soft_nms_high_threshold=0.9,
        post_process_top_k=100)

    localizer_pem = build_localizer(model_cfg)
    bsp_feature = torch.rand(8, 100, 32)
    reference_temporal_iou = torch.rand(8, 100)
    losses = localizer_pem(bsp_feature, reference_temporal_iou)
    assert isinstance(losses, dict)

    # Test forward test
    tmin = torch.rand(100)
    tmax = torch.rand(100)
    tmin_score = torch.rand(100)
    tmax_score = torch.rand(100)

    video_meta = [
        dict(
            video_name='v_test',
            duration_second=100,
            duration_frame=1000,
            annotations=[{
                'segment': [0.3, 0.6],
                'label': 'Rock climbing'
            }],
            feature_frame=900)
    ]
    with torch.no_grad():
        for one_bsp_feature in bsp_feature:
            one_bsp_feature = one_bsp_feature.reshape(1, 100, 32)
            localizer_pem(
                one_bsp_feature,
                tmin=tmin,
                tmax=tmax,
                tmin_score=tmin_score,
                tmax_score=tmax_score,
                video_meta=video_meta,
                return_loss=False)


def test_bmn():
    model_cfg = dict(
        type='BMN',
        temporal_dim=100,
        boundary_ratio=0.5,
        num_samples=32,
        num_samples_per_bin=3,
        feat_dim=400,
        soft_nms_alpha=0.4,
        soft_nms_low_threshold=0.5,
        soft_nms_high_threshold=0.9,
        post_process_top_k=100)
    if torch.cuda.is_available():
        localizer_bmn = build_localizer(model_cfg).cuda()
        raw_feature = torch.rand(8, 400, 100).cuda()
        gt_bbox = np.array([[[0.1, 0.3], [0.375, 0.625]]] * 8)
        losses = localizer_bmn(raw_feature, gt_bbox)
        assert isinstance(losses, dict)

        # Test forward test
        video_meta = [
            dict(
                video_name='v_test',
                duration_second=100,
                duration_frame=960,
                feature_frame=960)
        ]
        with torch.no_grad():
            one_raw_feature = torch.rand(1, 400, 100).cuda()
            localizer_bmn(
                one_raw_feature,
                gt_bbox=None,
                video_meta=video_meta,
                return_loss=False)
    else:
        localizer_bmn = build_localizer(model_cfg)
        raw_feature = torch.rand(8, 400, 100)
        gt_bbox = torch.Tensor([[[0.1, 0.3], [0.375, 0.625]]] * 8)
        losses = localizer_bmn(raw_feature, gt_bbox)
        assert isinstance(losses, dict)

        # Test forward test
        video_meta = [
            dict(
                video_name='v_test',
                duration_second=100,
                duration_frame=960,
                feature_frame=960)
        ]
        with torch.no_grad():
            one_raw_feature = torch.rand(1, 400, 100)
            localizer_bmn(
                one_raw_feature,
                gt_bbox=None,
                video_meta=video_meta,
                return_loss=False)


def test_post_processing():
    # test with multiple results
    result = np.array([[0., 1., 1., 1., 0.5, 0.5], [0., 0.4, 1., 1., 0.4, 0.4],
                       [0., 0.95, 1., 1., 0.6, 0.6]])
    video_info = dict(
        video_name='v_test',
        duration_second=100,
        duration_frame=960,
        feature_frame=960)
    proposal_list = post_processing(result, video_info, 0.75, 0.65, 0.9, 2, 16)
    assert isinstance(proposal_list[0], dict)
    assert proposal_list[0]['score'] == 0.6
    assert proposal_list[0]['segment'] == [0., 95.0]
    assert isinstance(proposal_list[1], dict)
    assert proposal_list[1]['score'] == 0.4
    assert proposal_list[1]['segment'] == [0., 40.0]

    # test with only result
    result = np.array([[0., 1., 1., 1., 0.5, 0.5]])
    video_info = dict(
        video_name='v_test',
        duration_second=100,
        duration_frame=960,
        feature_frame=960)
    proposal_list = post_processing(result, video_info, 0.75, 0.65, 0.9, 1, 16)
    assert isinstance(proposal_list[0], dict)
    assert proposal_list[0]['score'] == 0.5
    assert proposal_list[0]['segment'] == [0., 100.0]


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
