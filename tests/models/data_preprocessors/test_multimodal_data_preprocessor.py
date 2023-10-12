# Copyright (c) OpenMMLab. All rights reserved.
import copy

import pytest
import torch
from numpy.testing import assert_array_equal

from mmaction.models import MultiModalDataPreprocessor
from mmaction.structures import ActionDataSample
from mmaction.utils import register_all_modules


def generate_dummy_data(batch_size, input_keys, input_shapes):
    data = dict()
    data['data_samples'] = [
        ActionDataSample().set_gt_label(2) for _ in range(batch_size)
    ]
    data['inputs'] = dict()
    for key, shape in zip(input_keys, input_shapes):
        data['inputs'][key] = [
            torch.randint(0, 255, shape) for _ in range(batch_size)
        ]

    return data


def test_multimodal_data_preprocessor():
    with pytest.raises(AssertionError):
        MultiModalDataPreprocessor(
            preprocessors=dict(imgs=dict(format_shape='NCTHW')))

    register_all_modules()
    data_keys = ('imgs', 'heatmap_imgs')
    data_shapes = ((1, 3, 8, 224, 224), (1, 17, 32, 64, 64))
    raw_data = generate_dummy_data(2, data_keys, data_shapes)

    psr = MultiModalDataPreprocessor(
        preprocessors=dict(
            imgs=dict(
                type='ActionDataPreprocessor',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                format_shape='NCTHW'),
            heatmap_imgs=dict(type='ActionDataPreprocessor')))

    data = psr(copy.deepcopy(raw_data))
    assert data['inputs']['imgs'].shape == (2, 1, 3, 8, 224, 224)
    assert data['inputs']['heatmap_imgs'].shape == (2, 1, 17, 32, 64, 64)
    psr_imgs = psr.preprocessors['imgs']
    assert_array_equal(data['inputs']['imgs'][0],
                       (raw_data['inputs']['imgs'][0] - psr_imgs.mean) /
                       psr_imgs.std)
    assert_array_equal(data['inputs']['imgs'][1],
                       (raw_data['inputs']['imgs'][1] - psr_imgs.mean) /
                       psr_imgs.std)
    assert_array_equal(data['inputs']['heatmap_imgs'][0],
                       raw_data['inputs']['heatmap_imgs'][0])
    assert_array_equal(data['inputs']['heatmap_imgs'][1],
                       raw_data['inputs']['heatmap_imgs'][1])

    data_keys = ('imgs_2D', 'imgs_3D')
    data_shapes = ((1, 3, 224, 224), (1, 3, 8, 224, 224))
    raw_data = generate_dummy_data(2, data_keys, data_shapes)

    psr = MultiModalDataPreprocessor(
        preprocessors=dict(
            imgs_2D=dict(
                type='ActionDataPreprocessor',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                format_shape='NCHW'),
            imgs_3D=dict(
                type='ActionDataPreprocessor',
                mean=[127.5, 127.5, 127.5],
                std=[57.5, 57.5, 57.5],
                format_shape='NCTHW')))

    data = psr(copy.deepcopy(raw_data))
    assert data['inputs']['imgs_2D'].shape == (2, 1, 3, 224, 224)
    assert data['inputs']['imgs_3D'].shape == (2, 1, 3, 8, 224, 224)
    psr_imgs2d = psr.preprocessors['imgs_2D']
    psr_imgs3d = psr.preprocessors['imgs_3D']
    assert_array_equal(data['inputs']['imgs_2D'][0],
                       (raw_data['inputs']['imgs_2D'][0] - psr_imgs2d.mean) /
                       psr_imgs2d.std)
    assert_array_equal(data['inputs']['imgs_2D'][1],
                       (raw_data['inputs']['imgs_2D'][1] - psr_imgs2d.mean) /
                       psr_imgs2d.std)
    assert_array_equal(data['inputs']['imgs_3D'][0],
                       (raw_data['inputs']['imgs_3D'][0] - psr_imgs3d.mean) /
                       psr_imgs3d.std)
    assert_array_equal(data['inputs']['imgs_3D'][1],
                       (raw_data['inputs']['imgs_3D'][1] - psr_imgs3d.mean) /
                       psr_imgs3d.std)
