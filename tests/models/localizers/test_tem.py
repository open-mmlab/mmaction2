# Copyright (c) OpenMMLab. All rights reserved.
import platform

import numpy as np
import pytest
import torch
from mmcv.transforms import to_tensor
from mmengine.structures import InstanceData

from mmaction.registry import MODELS
from mmaction.structures import ActionDataSample
from mmaction.testing import get_localizer_cfg
from mmaction.utils import register_all_modules

register_all_modules()


def get_localization_data_sample():
    gt_bbox = np.array([[0.1, 0.3], [0.375, 0.625]])
    data_sample = ActionDataSample()
    instance_data = InstanceData()
    instance_data['gt_bbox'] = to_tensor(gt_bbox)
    data_sample.gt_instances = instance_data
    data_sample.set_metainfo(
        dict(
            video_name='v_test',
            duration_second=100,
            duration_frame=960,
            feature_frame=960))
    return data_sample


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_tem():
    model_cfg = get_localizer_cfg(
        'bsn/bsn_tem_1xb16-400x100-20e_activitynet-feature.py')

    localizer_tem = MODELS.build(model_cfg.model)
    raw_feature = torch.rand(8, 400, 100)
    # gt_bbox = torch.Tensor([[[1.0, 3.0], [3.0, 5.0]]] * 8)
    data_samples = [get_localization_data_sample()] * 8
    losses = localizer_tem(raw_feature, data_samples, mode='loss')
    assert isinstance(losses, dict)

    # Test forward predict
    with torch.no_grad():
        for one_raw_feature in raw_feature:
            one_raw_feature = one_raw_feature.reshape(1, 400, 100)
            data_samples = [get_localization_data_sample()]
            localizer_tem(one_raw_feature, data_samples, mode='predict')
