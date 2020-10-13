import numpy as np
import pytest
import torch
from mmcv.parallel import DataContainer as DC

from mmaction.datasets.pipelines import (Collect, FormatAudioShape,
                                         FormatShape, ImageToTensor,
                                         ToDataContainer, ToTensor, Transpose)


def check_keys_contain(result_keys, target_keys):
    """Check if all elements in target_keys is in result_keys."""
    return set(target_keys).issubset(set(result_keys))


def test_to_tensor():
    to_tensor = ToTensor(['str'])
    with pytest.raises(TypeError):
        # str cannot be converted to tensor
        results = dict(str='0')
        to_tensor(results)

    # convert tensor, numpy, squence, int, float to tensor
    target_keys = ['tensor', 'numpy', 'sequence', 'int', 'float']
    to_tensor = ToTensor(target_keys)
    original_results = dict(
        tensor=torch.randn(2, 3),
        numpy=np.random.randn(2, 3),
        sequence=list(range(10)),
        int=1,
        float=0.1)
    results = to_tensor(original_results)
    assert check_keys_contain(results.keys(), target_keys)
    for key in target_keys:
        assert isinstance(results[key], torch.Tensor)
        assert torch.equal(results[key].data, original_results[key])

    # Add an additional key which is not in keys.
    original_results = dict(
        tensor=torch.randn(2, 3),
        numpy=np.random.randn(2, 3),
        sequence=list(range(10)),
        int=1,
        float=0.1,
        str='test')
    results = to_tensor(original_results)
    assert check_keys_contain(results.keys(), target_keys)
    for key in target_keys:
        assert isinstance(results[key], torch.Tensor)
        assert torch.equal(results[key].data, original_results[key])

    assert repr(to_tensor) == to_tensor.__class__.__name__ + \
        f'(keys={target_keys})'


def test_to_data_container():
    # check user-defined fields
    fields = (dict(key='key1', stack=True), dict(key='key2'))
    to_data_container = ToDataContainer(fields=fields)
    target_keys = ['key1', 'key2']
    original_results = dict(key1=np.random.randn(10, 20), key2=['a', 'b'])
    results = to_data_container(original_results.copy())
    assert check_keys_contain(results.keys(), target_keys)
    for key in target_keys:
        assert isinstance(results[key], DC)
        assert np.all(results[key].data == original_results[key])
    assert results['key1'].stack
    assert not results['key2'].stack

    # Add an additional key which is not in keys.
    original_results = dict(
        key1=np.random.randn(10, 20), key2=['a', 'b'], key3='value3')
    results = to_data_container(original_results.copy())
    assert check_keys_contain(results.keys(), target_keys)
    for key in target_keys:
        assert isinstance(results[key], DC)
        assert np.all(results[key].data == original_results[key])
    assert results['key1'].stack
    assert not results['key2'].stack

    assert repr(to_data_container) == (
        to_data_container.__class__.__name__ + f'(fields={fields})')


def test_image_to_tensor():
    original_results = dict(imgs=np.random.randn(256, 256, 3))
    keys = ['imgs']
    image_to_tensor = ImageToTensor(keys)
    results = image_to_tensor(original_results)
    assert results['imgs'].shape == torch.Size([3, 256, 256])
    assert isinstance(results['imgs'], torch.Tensor)
    assert torch.equal(results['imgs'].data, original_results['imgs'])
    assert repr(image_to_tensor) == image_to_tensor.__class__.__name__ + \
        f'(keys={keys})'


def test_transpose():
    results = dict(imgs=np.random.randn(256, 256, 3))
    keys = ['imgs']
    order = [2, 0, 1]
    transpose = Transpose(keys, order)
    results = transpose(results)
    assert results['imgs'].shape == (3, 256, 256)
    assert repr(transpose) == transpose.__class__.__name__ + \
        f'(keys={keys}, order={order})'


def test_collect():
    inputs = dict(
        imgs=np.random.randn(256, 256, 3),
        label=[1],
        filename='test.txt',
        original_shape=(256, 256, 3),
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
        f'(keys={keys}, meta_keys={collect.meta_keys})'


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
    assert check_keys_contain(results.keys(), target_keys)

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
