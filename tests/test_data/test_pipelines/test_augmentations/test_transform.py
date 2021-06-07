import copy

import numpy as np
import pytest
from mmcv.utils import assert_dict_has_keys
from numpy.testing import assert_array_almost_equal

from mmaction.datasets.pipelines import RandomRescale, RandomScale, Resize
from mmaction.datasets.pipelines.augmentations import PoseCompact


class TestTransform:

    @staticmethod
    def test_random_rescale():
        with pytest.raises(AssertionError):
            # scale_range must be a tuple of int
            RandomRescale(scale_range=224)

        with pytest.raises(AssertionError):
            # scale_range must be a tuple of int
            RandomRescale(scale_range=(224.0, 256.0))

        with pytest.raises(AssertionError):
            # scale_range[0] > scale_range[1], which is wrong
            RandomRescale(scale_range=(320, 256))

        with pytest.raises(AssertionError):
            # scale_range[0] <= 0, which is wrong
            RandomRescale(scale_range=(0, 320))

        target_keys = ['imgs', 'short_edge', 'img_shape']
        # There will be a slight difference because of rounding
        eps = 0.01
        imgs = list(np.random.rand(2, 256, 340, 3))
        results = dict(imgs=imgs, img_shape=(256, 340), modality='RGB')

        random_rescale = RandomRescale(scale_range=(300, 400))
        random_rescale_result = random_rescale(results)

        assert assert_dict_has_keys(random_rescale_result, target_keys)

        h, w = random_rescale_result['img_shape']

        # check rescale
        assert np.abs(h / 256 - w / 340) < eps
        assert 300 / 256 - eps <= h / 256 <= 400 / 256 + eps
        assert repr(random_rescale) == (f'{random_rescale.__class__.__name__}'
                                        f'(scale_range={(300, 400)}, '
                                        'interpolation=bilinear)')

    @staticmethod
    def test_resize():
        with pytest.raises(ValueError):
            # scale must be positive
            Resize(-0.5)

        with pytest.raises(TypeError):
            # scale must be tuple of int
            Resize('224')

        target_keys = [
            'imgs', 'img_shape', 'keep_ratio', 'scale_factor', 'modality'
        ]

        # test resize for flow images
        imgs = list(np.random.rand(2, 240, 320))
        kp = np.array([60, 60]).reshape([1, 1, 1, 2])
        results = dict(imgs=imgs, keypoint=kp, modality='Flow')
        resize = Resize(scale=(160, 80), keep_ratio=False)
        resize_results = resize(results)
        assert assert_dict_has_keys(resize_results, target_keys)
        assert np.all(resize_results['scale_factor'] == np.array(
            [.5, 1. / 3.], dtype=np.float32))
        assert resize_results['img_shape'] == (80, 160)
        kp = resize_results['keypoint'][0, 0, 0]
        assert_array_almost_equal(kp, np.array([30, 20]))

        # scale with -1 to indicate np.inf
        imgs = list(np.random.rand(2, 240, 320, 3))
        results = dict(imgs=imgs, modality='RGB')
        results['gt_bboxes'] = np.array([[0, 0, 320, 240]])
        results['proposals'] = np.array([[0, 0, 320, 240]])
        resize = Resize(scale=(-1, 256), keep_ratio=True)
        resize_results = resize(results)
        assert assert_dict_has_keys(resize_results, target_keys)
        assert np.all(resize_results['scale_factor'] == np.array(
            [341 / 320, 256 / 240], dtype=np.float32))
        assert resize_results['img_shape'] == (256, 341)

        # scale with a normal tuple (320, 320) to indicate np.inf
        imgs = list(np.random.rand(2, 240, 320, 3))
        results = dict(imgs=imgs, modality='RGB')
        resize = Resize(scale=(320, 320), keep_ratio=False)
        resize_results = resize(results)
        assert assert_dict_has_keys(resize_results, target_keys)
        assert np.all(resize_results['scale_factor'] == np.array(
            [1, 320 / 240], dtype=np.float32))
        assert resize_results['img_shape'] == (320, 320)

        # scale with a normal tuple (341, 256) to indicate np.inf
        imgs = list(np.random.rand(2, 240, 320, 3))
        results = dict(imgs=imgs, modality='RGB')
        resize = Resize(scale=(341, 256), keep_ratio=False)
        resize_results = resize(results)
        assert assert_dict_has_keys(resize_results, target_keys)
        assert np.all(resize_results['scale_factor'] == np.array(
            [341 / 320, 256 / 240], dtype=np.float32))
        assert resize_results['img_shape'] == (256, 341)

        assert repr(resize) == (
            resize.__class__.__name__ +
            f'(scale={(341, 256)}, keep_ratio={False}, ' +
            f'interpolation=bilinear, lazy={False})')

    @staticmethod
    def test_random_scale():
        scales = ((200, 64), (250, 80))
        with pytest.raises(ValueError):
            RandomScale(scales, 'unsupport')

        with pytest.raises(ValueError):
            random_scale = RandomScale([(800, 256), (1000, 320), (800, 320)])
            random_scale({})

        imgs = list(np.random.rand(2, 340, 256, 3))
        results = dict(imgs=imgs, img_shape=(340, 256))

        results_ = copy.deepcopy(results)
        random_scale_range = RandomScale(scales)
        results_ = random_scale_range(results_)
        assert 200 <= results_['scale'][0] <= 250
        assert 64 <= results_['scale'][1] <= 80

        results_ = copy.deepcopy(results)
        random_scale_value = RandomScale(scales, 'value')
        results_ = random_scale_value(results_)
        assert results_['scale'] in scales

        random_scale_single = RandomScale([(200, 64)])
        results_ = copy.deepcopy(results)
        results_ = random_scale_single(results_)
        assert results_['scale'] == (200, 64)

        assert repr(random_scale_range) == (
            f'{random_scale_range.__class__.__name__}'
            f'(scales={((200, 64), (250, 80))}, '
            'mode=range)')


class TestPoseCompact:

    @staticmethod
    def test_pose_compact():
        results = {}
        results['img_shape'] = (100, 100)
        fake_kp = np.zeros([1, 4, 2, 2])
        fake_kp[:, :, 0] = [10, 10]
        fake_kp[:, :, 1] = [90, 90]
        results['keypoint'] = fake_kp

        pose_compact = PoseCompact(
            padding=0, threshold=0, hw_ratio=None, allow_imgpad=False)
        inp = copy.deepcopy(results)
        ret = pose_compact(inp)
        assert ret['img_shape'] == (80, 80)
        assert str(pose_compact) == (
            'PoseCompact(padding=0, threshold=0, hw_ratio=None, '
            'allow_imgpad=False)')

        pose_compact = PoseCompact(
            padding=0.3, threshold=0, hw_ratio=None, allow_imgpad=False)
        inp = copy.deepcopy(results)
        ret = pose_compact(inp)
        assert ret['img_shape'] == (100, 100)

        pose_compact = PoseCompact(
            padding=0.3, threshold=0, hw_ratio=None, allow_imgpad=True)
        inp = copy.deepcopy(results)
        ret = pose_compact(inp)
        assert ret['img_shape'] == (104, 104)

        pose_compact = PoseCompact(
            padding=0, threshold=100, hw_ratio=None, allow_imgpad=False)
        inp = copy.deepcopy(results)
        ret = pose_compact(inp)
        assert ret['img_shape'] == (100, 100)

        pose_compact = PoseCompact(
            padding=0, threshold=0, hw_ratio=0.75, allow_imgpad=True)
        inp = copy.deepcopy(results)
        ret = pose_compact(inp)
        assert ret['img_shape'] == (80, 106)
