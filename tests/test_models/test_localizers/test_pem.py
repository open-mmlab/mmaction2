import torch

from mmaction.models import build_localizer


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
