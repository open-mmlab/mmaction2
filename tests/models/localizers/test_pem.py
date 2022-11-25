# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
import torch
from mmengine.structures import InstanceData

from mmaction.registry import MODELS
from mmaction.structures import ActionDataSample
from mmaction.testing import get_localizer_cfg
from mmaction.utils import register_all_modules

register_all_modules()


def get_localization_data_sample():
    bsp_feature = torch.rand(100, 32)
    reference_temporal_iou = torch.rand(100)
    data_sample = ActionDataSample()
    instance_data = InstanceData()
    instance_data['bsp_feature'] = bsp_feature
    instance_data['reference_temporal_iou'] = reference_temporal_iou
    data_sample.gt_instances = instance_data
    return data_sample


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_pem():
    model_cfg = get_localizer_cfg(
        'bsn/bsn_pem_1xb16-400x100-20e_activitynet-feature.py')

    localizer_pem = MODELS.build(model_cfg.model)
    raw_features = [torch.rand(100, 32)] * 8
    data_samples = [get_localization_data_sample()] * 8
    losses = localizer_pem(raw_features, data_samples, mode='loss')
    assert isinstance(losses, dict)

    # Test forward predict
    tmin = torch.rand(100)
    tmax = torch.rand(100)
    tmin_score = torch.rand(100)
    tmax_score = torch.rand(100)

    video_meta = dict(
        video_name='v_test',
        duration_second=100,
        duration_frame=1000,
        annotations=[{
            'segment': [0.3, 0.6],
            'label': 'Rock climbing'
        }],
        feature_frame=900)

    with torch.no_grad():
        raw_feature = [torch.rand(100, 32)]
        data_sample = get_localization_data_sample()
        data_sample.set_metainfo(video_meta)
        gt_instances = data_sample.gt_instances
        gt_instances['tmin'] = tmin
        gt_instances['tmax'] = tmax
        gt_instances['tmin_score'] = tmin_score
        gt_instances['tmax_score'] = tmax_score
        data_samples = [data_sample]

        localizer_pem(raw_feature, data_samples, mode='predict')

    # Test forward tensor
    with torch.no_grad():
        raw_feature = [torch.rand(100, 32)]
        localizer_pem(raw_feature, data_samples=None, mode='tensor')
