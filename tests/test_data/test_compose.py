import numpy as np
import pytest

from mmaction.datasets.pipelines import Compose, ImageToTensor


def check_keys_equal(result_keys, target_keys):
    """Check if all elements in target_keys is in result_keys."""
    return set(target_keys) == set(result_keys)


def test_compose():
    with pytest.raises(TypeError):
        # transform must be callable or a dict
        Compose('LoadImage')

    target_keys = ['img', 'img_meta']

    # test Compose given a data pipeline
    img = np.random.randn(256, 256, 3)
    results = dict(img=img, abandoned_key=None, img_name='test_image.png')
    test_pipeline = [
        dict(type='Collect', keys=['img'], meta_keys=['img_name']),
        dict(type='ImageToTensor', keys=['img'])
    ]
    compose = Compose(test_pipeline)
    compose_results = compose(results)
    assert check_keys_equal(compose_results.keys(), target_keys)
    assert check_keys_equal(compose_results['img_meta'].data.keys(),
                            ['img_name'])

    # test Compose when forward data is None
    results = None
    image_to_tensor = ImageToTensor(keys=[])
    test_pipeline = [image_to_tensor]
    compose = Compose(test_pipeline)
    compose_results = compose(results)
    assert compose_results is None

    assert repr(compose) == compose.__class__.__name__ + \
        '(\n    {}\n)'.format(image_to_tensor)
