import numpy as np
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


def test_post_processing():
    # test with multiple results
    result = np.array([[0., 1., 1., 1., 0.5, 0.5], [0., 0.4, 1., 1., 0.4, 0.4],
                       [0., 0.95, 1., 1., 0.6, 0.6]])
    video_info = dict(
        video_name='v_test',
        duration_second=100,
        duration_frame=960,
        feature_frame=960)
    proposal_list = post_processing(result, video_info, 0.75, 0.65, 0.9, 2)
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
    proposal_list = post_processing(result, video_info, 0.75, 0.65, 0.9, 1)
    assert isinstance(proposal_list[0], dict)
    assert proposal_list[0]['score'] == 0.5
    assert proposal_list[0]['segment'] == [0., 100.0]
