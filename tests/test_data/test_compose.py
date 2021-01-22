import numpy as np
import pytest
from mmcv.utils import assert_keys_equal

from mmaction.datasets.pipelines import Compose, ImageToTensor


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
