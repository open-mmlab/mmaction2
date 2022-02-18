# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
import torch

from mmaction.models import build_localizer
from ..base import get_localizer_cfg


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_tem():
    model_cfg = get_localizer_cfg(
        'bsn/bsn_tem_400x100_1x16_20e_activitynet_feature.py')

    localizer_tem = build_localizer(model_cfg.model)
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
