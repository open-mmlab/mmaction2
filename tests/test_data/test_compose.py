# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
from mmcv.utils import assert_keys_equal, digit_version

from mmaction.datasets.pipelines import Compose, ImageToTensor

try:
    import torchvision
    torchvision_ok = False
    if digit_version(torchvision.__version__) >= digit_version('0.8.0'):
        torchvision_ok = True
except (ImportError, ModuleNotFoundError):
    torchvision_ok = False


def test_compose():
    with pytest.raises(TypeError):
        # transform must be callable or a dict
        Compose('LoadImage')

    target_keys = ['img', 'img_metas']

    # test Compose given a data pipeline
    img = np.random.randn(256, 256, 3)
    results = dict(img=img, abandoned_key=None, img_name='test_image.png')
    test_pipeline = [
        dict(type='Collect', keys=['img'], meta_keys=['img_name']),
        dict(type='ImageToTensor', keys=['img'])
    ]
    compose = Compose(test_pipeline)
    compose_results = compose(results)
    assert assert_keys_equal(compose_results.keys(), target_keys)
    assert assert_keys_equal(compose_results['img_metas'].data.keys(),
                             ['img_name'])

    # test Compose when forward data is None
    results = None
    image_to_tensor = ImageToTensor(keys=[])
    test_pipeline = [image_to_tensor]
    compose = Compose(test_pipeline)
    compose_results = compose(results)
    assert compose_results is None

    assert repr(compose) == compose.__class__.__name__ + \
        f'(\n    {image_to_tensor}\n)'


@pytest.mark.skipif(
    not torchvision_ok, reason='torchvision >= 0.8.0 is required')
def test_compose_support_torchvision():
    target_keys = ['imgs', 'img_metas']

    # test Compose given a data pipeline
    imgs = [np.random.randn(256, 256, 3)] * 8
    results = dict(
        imgs=imgs,
        abandoned_key=None,
        img_name='test_image.png',
        clip_len=8,
        num_clips=1)
    test_pipeline = [
        dict(type='torchvision.Grayscale', num_output_channels=3),
        dict(type='FormatShape', input_format='NCTHW'),
        dict(type='Collect', keys=['imgs'], meta_keys=['img_name']),
        dict(type='ToTensor', keys=['imgs'])
    ]
    compose = Compose(test_pipeline)
    compose_results = compose(results)
    assert assert_keys_equal(compose_results.keys(), target_keys)
    assert assert_keys_equal(compose_results['img_metas'].data.keys(),
                             ['img_name'])
