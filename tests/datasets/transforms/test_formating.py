# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
from mmcv.utils import assert_dict_has_keys

from mmaction.datasets.transforms import (FormatAudioShape, FormatGCNInput,
                                          FormatShape, Transpose)


def test_transpose():
    results = dict(imgs=np.random.randn(256, 256, 3))
    keys = ['imgs']
    order = [2, 0, 1]
    transpose = Transpose(keys, order)
    results = transpose(results)
    assert results['imgs'].shape == (3, 256, 256)
    assert repr(transpose) == transpose.__class__.__name__ + \
        f'(keys={keys}, order={order})'


def test_format_shape():
    with pytest.raises(ValueError):
        # invalid input format
        FormatShape('NHWC')

    # 'NCHW' input format
    results = dict(
        imgs=np.random.randn(3, 224, 224, 3), num_clips=1, clip_len=3)
    format_shape = FormatShape('NCHW')
    assert format_shape(results)['input_shape'] == (3, 3, 224, 224)

    # `NCTHW` input format with num_clips=1, clip_len=3
    results = dict(
        imgs=np.random.randn(3, 224, 224, 3), num_clips=1, clip_len=3)
    format_shape = FormatShape('NCTHW')
    assert format_shape(results)['input_shape'] == (1, 3, 3, 224, 224)

    # `NCTHW` input format with num_clips=2, clip_len=3
    results = dict(
        imgs=np.random.randn(18, 224, 224, 3), num_clips=2, clip_len=3)
    assert format_shape(results)['input_shape'] == (6, 3, 3, 224, 224)
    target_keys = ['imgs', 'input_shape']
    assert assert_dict_has_keys(results, target_keys)

    assert repr(format_shape) == format_shape.__class__.__name__ + \
        "(input_format='NCTHW')"

    # 'NPTCHW' input format
    results = dict(
        imgs=np.random.randn(72, 224, 224, 3),
        num_clips=9,
        clip_len=1,
        num_proposals=8)
    format_shape = FormatShape('NPTCHW')
    assert format_shape(results)['input_shape'] == (8, 9, 3, 224, 224)


def test_format_audio_shape():
    with pytest.raises(ValueError):
        # invalid input format
        FormatAudioShape('XXXX')

    # 'NCTF' input format
    results = dict(audios=np.random.randn(3, 128, 8))
    format_shape = FormatAudioShape('NCTF')
    assert format_shape(results)['input_shape'] == (3, 1, 128, 8)
    assert repr(format_shape) == format_shape.__class__.__name__ + \
        "(input_format='NCTF')"


def test_format_gcn_input():
    with pytest.raises(ValueError):
        # invalid input format
        FormatGCNInput('XXXX')

    # 'NCTVM' input format
    results = dict(
        keypoint=np.random.randn(2, 300, 17, 2),
        keypoint_score=np.random.randn(2, 300, 17))
    format_shape = FormatGCNInput('NCTVM', num_person=2)
    assert format_shape(results)['input_shape'] == (3, 300, 17, 2)
    assert repr(format_shape) == format_shape.__class__.__name__ + \
        '(input_format=NCTVM, num_person=%d)' % 2

    # test real num_person < 2
    results = dict(
        keypoint=np.random.randn(1, 300, 17, 2),
        keypoint_score=np.random.randn(1, 300, 17))
    assert format_shape(results)['input_shape'] == (3, 300, 17, 2)
    assert repr(format_shape) == format_shape.__class__.__name__ + \
        '(input_format=NCTVM, num_person=%d)' % 2

    # test real num_person > 2
    results = dict(
        keypoint=np.random.randn(3, 300, 17, 2),
        keypoint_score=np.random.randn(3, 300, 17))
    assert format_shape(results)['input_shape'] == (3, 300, 17, 2)
    assert repr(format_shape) == format_shape.__class__.__name__ + \
        '(input_format=NCTVM, num_person=%d)' % 2
