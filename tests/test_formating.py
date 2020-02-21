import numpy as np
import pytest
import torch

from mmaction.datasets.pipelines import (Collect, FormatShape, ImageToTensor,
                                         ToTensor, Transpose)


def check_keys_contain(result_keys, target_keys):
    """Check if all elements in target_keys is in result_keys."""
    return set(target_keys).issubset(set(result_keys))


def test_to_tensor():
    to_tensor = ToTensor(['str'])
    with pytest.raises(TypeError):
        results = dict(str='0')
        to_tensor(results)

    target_keys = ['tensor', 'numpy', 'sequence', 'int', 'float']
    to_tensor = ToTensor(target_keys)
    origin_results = dict(
        tensor=torch.randn(2, 3),
        numpy=np.random.randn(2, 3),
        sequence=list(range(10)),
        int=1,
        float=0.1)

    results = to_tensor(origin_results)
    assert check_keys_contain(results.keys(), target_keys)
    for key in target_keys:
        assert isinstance(results[key], torch.Tensor)
        assert torch.equal(results[key].data, origin_results[key])

    # Add an additional key which is not in keys.
    origin_results = dict(
        tensor=torch.randn(2, 3),
        numpy=np.random.randn(2, 3),
        sequence=list(range(10)),
        int=1,
        float=0.1,
        str='test')
    results = to_tensor(origin_results)
    assert check_keys_contain(results.keys(), target_keys)
    for key in target_keys:
        assert isinstance(results[key], torch.Tensor)
        assert torch.equal(results[key].data, origin_results[key])

    assert repr(to_tensor) == to_tensor.__class__.__name__ + \
        '(keys={})'.format(target_keys)


def test_image_to_tensor():
    origin_results = dict(imgs=np.random.randn(256, 256, 3))
    keys = ['imgs']
    image_to_tensor = ImageToTensor(keys)
    results = image_to_tensor(origin_results)
    assert results['imgs'].shape == torch.Size([3, 256, 256])
    assert isinstance(results['imgs'], torch.Tensor)
    assert torch.equal(results['imgs'].data, origin_results['imgs'])

    assert repr(image_to_tensor) == image_to_tensor.__class__.__name__ + \
        '(keys={})'.format(keys)


def test_transpose():
    results = dict(imgs=np.random.randn(256, 256, 3))
    keys = ['imgs']
    order = [2, 0, 1]
    transpose = Transpose(keys, order)
    results = transpose(results)
    assert results['imgs'].shape == (3, 256, 256)

    assert repr(transpose) == transpose.__class__.__name__ + \
        '(keys={}, order={})'.format(keys, order)


def test_collect():
    inputs = dict(
        imgs=np.random.randn(256, 256, 3),
        label=[1],
        filename='test.txt',
        ori_shape=(256, 256, 3),
        img_shape=(256, 256, 3),
        pad_shape=(256, 256, 3),
        flip_direction='vertical',
        img_norm_cfg=dict(to_bgr=False))
    keys = ['imgs', 'label']
    collect = Collect(keys)
    results = collect(inputs)
    assert sorted(list(results.keys())) == sorted(
        ['imgs', 'label', 'img_meta'])
    inputs.pop('imgs')
    assert set(results['img_meta'].data.keys()) == set(inputs.keys())
    for key in results['img_meta'].data:
        assert results['img_meta'].data[key] == inputs[key]

    assert repr(collect) == collect.__class__.__name__ + \
        '(keys={}, meta_keys={})'.format(keys, collect.meta_keys)


def test_format_shape():
    with pytest.raises(ValueError):
        FormatShape('NHWC')

    results = dict(
        imgs=np.random.randn(3, 224, 224, 3), num_clips=1, clip_len=3)
    format_shape = FormatShape('NCHW')
    assert format_shape(results)['input_shape'] == (3, 3, 224, 224)

    results = dict(
        imgs=np.random.randn(3, 224, 224, 3), num_clips=1, clip_len=3)
    format_shape = FormatShape('NCTHW')
    assert format_shape(results)['input_shape'] == (1, 3, 3, 224, 224)

    results = dict(
        imgs=np.random.randn(18, 224, 224, 3), num_clips=2, clip_len=3)
    assert format_shape(results)['input_shape'] == (6, 3, 3, 224, 224)

    target_keys = ['imgs', 'input_shape']
    assert check_keys_contain(results.keys(), target_keys)

    assert repr(format_shape) == format_shape.__class__.__name__ + \
        '(input_format={})'.format('NCTHW')
