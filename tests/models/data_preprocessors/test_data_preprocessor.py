# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy

import pytest
import torch
from numpy.testing import assert_array_equal

from mmaction.models import ActionDataPreprocessor
from mmaction.structures import ActionDataSample
from mmaction.utils import register_all_modules


def generate_dummy_data(batch_size, input_shape):
    data = {
        'inputs':
        [torch.randint(0, 255, input_shape) for _ in range(batch_size)],
        'data_samples':
        [ActionDataSample().set_gt_label(2) for _ in range(batch_size)]
    }
    return data


def test_data_preprocessor():
    with pytest.raises(ValueError):
        ActionDataPreprocessor(
            mean=[1, 1], std=[0, 0], format_shape='NCTHW_Heatmap')
    with pytest.raises(ValueError):
        psr = ActionDataPreprocessor(format_shape='NCTHW_Heatmap', to_rgb=True)
        psr(generate_dummy_data(1, (3, 224, 224)))

    raw_data = generate_dummy_data(2, (1, 3, 8, 224, 224))
    psr = ActionDataPreprocessor(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCTHW')
    data = psr(deepcopy(raw_data))
    assert data['inputs'].shape == (2, 1, 3, 8, 224, 224)
    assert_array_equal(data['inputs'][0],
                       (raw_data['inputs'][0] - psr.mean) / psr.std)
    assert_array_equal(data['inputs'][1],
                       (raw_data['inputs'][1] - psr.mean) / psr.std)

    psr = ActionDataPreprocessor(format_shape='NCTHW', to_rgb=True)
    data = psr(deepcopy(raw_data))
    assert data['inputs'].shape == (2, 1, 3, 8, 224, 224)
    assert_array_equal(data['inputs'][0], raw_data['inputs'][0][:, [2, 1, 0]])
    assert_array_equal(data['inputs'][1], raw_data['inputs'][1][:, [2, 1, 0]])

    register_all_modules()
    psr = ActionDataPreprocessor(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCTHW',
        blending=dict(type='MixupBlending', num_classes=5))
    data = psr(deepcopy(raw_data), training=True)
    assert data['data_samples'][0].gt_label.shape == (5, )
    assert data['data_samples'][1].gt_label.shape == (5, )

    raw_data = generate_dummy_data(2, (1, 3, 224, 224))
    psr = ActionDataPreprocessor(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCHW',
        to_rgb=True)
    data = psr(deepcopy(raw_data))
    assert_array_equal(data['inputs'][0],
                       (raw_data['inputs'][0][:, [2, 1, 0]] - psr.mean) /
                       psr.std)
    assert_array_equal(data['inputs'][1],
                       (raw_data['inputs'][1][:, [2, 1, 0]] - psr.mean) /
                       psr.std)

    psr = ActionDataPreprocessor()
    data = psr(deepcopy(raw_data))
    assert data['inputs'].shape == (2, 1, 3, 224, 224)
    assert_array_equal(data['inputs'][0], raw_data['inputs'][0])
    assert_array_equal(data['inputs'][1], raw_data['inputs'][1])

    raw_2d_data = generate_dummy_data(2, (3, 224, 224))
    raw_3d_data = generate_dummy_data(2, (1, 3, 8, 224, 224))
    raw_data = (raw_2d_data, raw_3d_data)

    psr = ActionDataPreprocessor(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='MIX2d3d')
    data = psr(raw_data)
    assert_array_equal(data[0]['inputs'][0],
                       (raw_2d_data['inputs'][0] - psr.mean.view(-1, 1, 1)) /
                       psr.std.view(-1, 1, 1))
    assert_array_equal(data[0]['inputs'][1],
                       (raw_2d_data['inputs'][1] - psr.mean.view(-1, 1, 1)) /
                       psr.std.view(-1, 1, 1))
    assert_array_equal(data[1]['inputs'][0],
                       (raw_3d_data['inputs'][0] - psr.mean) / psr.std)
    assert_array_equal(data[1]['inputs'][1],
                       (raw_3d_data['inputs'][1] - psr.mean) / psr.std)

    raw_data = generate_dummy_data(2, (77, ))
    psr = ActionDataPreprocessor(to_float32=False)
    data = psr(raw_data)
    assert data['inputs'].dtype == raw_data['inputs'][0].dtype
