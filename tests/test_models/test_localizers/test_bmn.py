# Copyright (c) OpenMMLab. All rights reserved.
import platform

import numpy as np
import pytest
import torch

from mmaction.models import build_localizer
from ..base import get_localizer_cfg


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_bmn_train():
    model_cfg = get_localizer_cfg(
        'bmn/bmn_400x100_2x8_9e_activitynet_feature.py')

    if torch.cuda.is_available():
        localizer_bmn = build_localizer(model_cfg.model).cuda()
        raw_feature = torch.rand(3, 400, 100).cuda()
        gt_bbox = np.array([[[0.1, 0.3], [0.375, 0.625]]] * 3)
        losses = localizer_bmn(raw_feature, gt_bbox)
        assert isinstance(losses, dict)

    else:
        localizer_bmn = build_localizer(model_cfg.model)
        raw_feature = torch.rand(3, 400, 100)
        gt_bbox = torch.Tensor([[[0.1, 0.3], [0.375, 0.625]]] * 3)
        losses = localizer_bmn(raw_feature, gt_bbox)
        assert isinstance(losses, dict)


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_bmn_test():
    model_cfg = get_localizer_cfg(
        'bmn/bmn_400x100_2x8_9e_activitynet_feature.py')

    if torch.cuda.is_available():
        localizer_bmn = build_localizer(model_cfg.model).cuda()
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
        localizer_bmn = build_localizer(model_cfg.model)
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
