# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmaction.models import build_localizer
from ..base import get_localizer_cfg


def test_pem():
    model_cfg = get_localizer_cfg(
        'bsn/bsn_pem_400x100_1x16_20e_activitynet_feature.py')

    localizer_pem = build_localizer(model_cfg.model)
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
