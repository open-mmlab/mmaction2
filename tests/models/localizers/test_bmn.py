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
def test_bmn_loss():
    model_cfg = get_localizer_cfg(
        'bmn/bmn_2xb8-400x100-9e_activitynet-feature.py')

    if 0 and torch.cuda.is_available():
        raw_feature = [torch.rand(400, 100).cuda()]
        data_samples = [get_localization_data_sample()]
        localizer_bmn = MODELS.build(model_cfg.model).cuda()
        losses = localizer_bmn(raw_feature, data_samples, mode='loss')
        assert isinstance(losses, dict)

    else:
        raw_feature = [torch.rand(400, 100)]
        data_samples = [get_localization_data_sample()]
        localizer_bmn = MODELS.build(model_cfg.model)
        losses = localizer_bmn(raw_feature, data_samples, mode='loss')
        assert isinstance(losses, dict)


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_bmn_predict():
    model_cfg = get_localizer_cfg(
        'bmn/bmn_2xb8-400x100-9e_activitynet-feature.py')

    if 0 and torch.cuda.is_available():
        localizer_bmn = MODELS.build(model_cfg.model).cuda()
        data_samples = [get_localization_data_sample()]

        with torch.no_grad():
            one_raw_feature = [torch.rand(400, 100).cuda()]
            localizer_bmn(one_raw_feature, data_samples, mode='predict')
    else:
        localizer_bmn = MODELS.build(model_cfg.model)
        data_samples = [get_localization_data_sample()]
        with torch.no_grad():
            one_raw_feature = [torch.rand(400, 100)]
            localizer_bmn(one_raw_feature, data_samples, mode='predict')


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_bmn_tensor():
    model_cfg = get_localizer_cfg(
        'bmn/bmn_2xb8-400x100-9e_activitynet-feature.py')

    if 0 and torch.cuda.is_available():
        localizer_bmn = MODELS.build(model_cfg.model).cuda()

        with torch.no_grad():
            one_raw_feature = [torch.rand(400, 100).cuda()]
            localizer_bmn(one_raw_feature, data_samples=None, mode='tensor')
    else:
        localizer_bmn = MODELS.build(model_cfg.model)
        with torch.no_grad():
            one_raw_feature = [torch.rand(400, 100)]
            localizer_bmn(one_raw_feature, data_samples=None, mode='tensor')
