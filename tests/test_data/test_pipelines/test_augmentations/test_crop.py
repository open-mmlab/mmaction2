# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
from mmcv.utils import assert_dict_has_keys

from mmaction.datasets.pipelines import (CenterCrop, MultiScaleCrop,
                                         RandomCrop, RandomResizedCrop,
                                         TenCrop, ThreeCrop)
from .base import check_crop


class TestCrops:

    @staticmethod
    def test_random_crop():
        with pytest.raises(TypeError):
            # size must be an int
            RandomCrop(size=(112, 112))
        with pytest.raises(AssertionError):
            # "size > height" or "size > width" is not allowed
            imgs = list(np.random.rand(2, 224, 341, 3))
            results = dict(imgs=imgs)
            random_crop = RandomCrop(size=320)
            random_crop(results)

        target_keys = ['imgs', 'crop_bbox', 'img_shape']

        # General case
        imgs = list(np.random.rand(2, 224, 341, 3))
        results = dict(imgs=imgs)
        random_crop = RandomCrop(size=224)
        results['gt_bboxes'] = np.array([[0, 0, 340, 224]])
        results['proposals'] = np.array([[0, 0, 340, 224]])
        kp = np.array([[160, 120], [160, 120]]).reshape([1, 1, 2, 2])
        results['keypoint'] = kp
        random_crop_result = random_crop(results)
        assert assert_dict_has_keys(random_crop_result, target_keys)
        assert check_crop(imgs, random_crop_result['imgs'],
                          results['crop_bbox'])
        h, w = random_crop_result['img_shape']
        assert h == w == 224

        # Test the case that no need for cropping
        imgs = list(np.random.rand(2, 224, 224, 3))
        results = dict(imgs=imgs)
        random_crop = RandomCrop(size=224)
        random_crop_result = random_crop(results)
        assert assert_dict_has_keys(random_crop_result, target_keys)
        assert check_crop(imgs, random_crop_result['imgs'],
                          results['crop_bbox'])
        h, w = random_crop_result['img_shape']
        assert h == w == 224

        # Test the one-side-equal case
        imgs = list(np.random.rand(2, 224, 225, 3))
        results = dict(imgs=imgs)
        random_crop = RandomCrop(size=224)
        random_crop_result = random_crop(results)
        assert assert_dict_has_keys(random_crop_result, target_keys)
        assert check_crop(imgs, random_crop_result['imgs'],
                          results['crop_bbox'])
        h, w = random_crop_result['img_shape']
        assert h == w == 224

        assert repr(random_crop) == (f'{random_crop.__class__.__name__}'
                                     f'(size={224}, lazy={False})')

    @staticmethod
    def test_random_resized_crop():
        with pytest.raises(TypeError):
            # area_range must be a tuple of float
            RandomResizedCrop(area_range=0.5)
        with pytest.raises(TypeError):
            # aspect_ratio_range must be a tuple of float
            RandomResizedCrop(area_range=(0.08, 1.0), aspect_ratio_range=0.1)

        target_keys = ['imgs', 'crop_bbox', 'img_shape']
        # There will be a slight difference because of rounding
        eps = 0.01
        imgs = list(np.random.rand(2, 256, 341, 3))
        results = dict(imgs=imgs)
        results['gt_bboxes'] = np.array([[0, 0, 340, 256]])
        results['proposals'] = np.array([[0, 0, 340, 256]])
        kp = np.array([[160, 120], [160, 120]]).reshape([1, 1, 2, 2])
        results['keypoint'] = kp

        with pytest.raises(AssertionError):
            # area_range[0] > area_range[1], which is wrong
            random_crop = RandomResizedCrop(area_range=(0.9, 0.7))
            random_crop(results)
        with pytest.raises(AssertionError):
            # 0 > area_range[0] and area_range[1] > 1, which is wrong
            random_crop = RandomResizedCrop(aspect_ratio_range=(-0.1, 2.0))
            random_crop(results)

        random_crop = RandomResizedCrop()
        random_crop_result = random_crop(results)
        assert assert_dict_has_keys(random_crop_result, target_keys)
        assert check_crop(imgs, random_crop_result['imgs'],
                          results['crop_bbox'])
        h, w = random_crop_result['img_shape']
        assert ((0.08 - eps <= h * w / 256 / 341)
                and (h * w / 256 / 341 <= 1 + eps))
        assert (3. / 4. - eps <= h / w) and (h / w - eps <= 4. / 3.)
        assert repr(random_crop) == (f'{random_crop.__class__.__name__}'
                                     f'(area_range={(0.08, 1.0)}, '
                                     f'aspect_ratio_range={(3 / 4, 4 / 3)}, '
                                     f'lazy={False})')

        random_crop = RandomResizedCrop(
            area_range=(0.9, 0.9), aspect_ratio_range=(10.0, 10.1))
        # Test fallback cases by very big area range
        imgs = list(np.random.rand(2, 256, 341, 3))
        results = dict(imgs=imgs)
        random_crop_result = random_crop(results)
        assert assert_dict_has_keys(random_crop_result, target_keys)
        assert check_crop(imgs, random_crop_result['imgs'],
                          results['crop_bbox'])
        h, w = random_crop_result['img_shape']
        assert h == w == 256

    @staticmethod
    def test_multi_scale_crop():
        with pytest.raises(TypeError):
            # input_size must be int or tuple of int
            MultiScaleCrop(0.5)

        with pytest.raises(TypeError):
            # input_size must be int or tuple of int
            MultiScaleCrop('224')

        with pytest.raises(TypeError):
            # scales must be tuple.
            MultiScaleCrop(
                224, scales=[
                    1,
                ])

        with pytest.raises(ValueError):
            # num_fix_crops must be in [5, 13]
            MultiScaleCrop(224, num_fixed_crops=6)

        target_keys = ['imgs', 'crop_bbox', 'img_shape', 'scales']

        # MultiScaleCrop with normal crops.
        imgs = list(np.random.rand(2, 256, 341, 3))
        results = dict(imgs=imgs)
        results['gt_bboxes'] = np.array([[0, 0, 340, 256]])
        results['proposals'] = np.array([[0, 0, 340, 256]])
        kp = np.array([[160, 120], [160, 120]]).reshape([1, 1, 2, 2])
        results['keypoint'] = kp
        config = dict(
            input_size=224,
            scales=(1, 0.8),
            random_crop=False,
            max_wh_scale_gap=0)
        multi_scale_crop = MultiScaleCrop(**config)
        multi_scale_crop_results = multi_scale_crop(results)
        assert assert_dict_has_keys(multi_scale_crop_results, target_keys)
        assert check_crop(imgs, multi_scale_crop_results['imgs'],
                          multi_scale_crop_results['crop_bbox'])
        assert multi_scale_crop_results['img_shape'] in [(256, 256),
                                                         (204, 204)]

        # MultiScaleCrop with more fixed crops.
        imgs = list(np.random.rand(2, 256, 341, 3))
        results = dict(imgs=imgs)
        config = dict(
            input_size=224,
            scales=(1, 0.8),
            random_crop=False,
            max_wh_scale_gap=0,
            num_fixed_crops=13)
        multi_scale_crop = MultiScaleCrop(**config)
        multi_scale_crop_results = multi_scale_crop(results)
        assert assert_dict_has_keys(multi_scale_crop_results, target_keys)
        assert check_crop(imgs, multi_scale_crop_results['imgs'],
                          multi_scale_crop_results['crop_bbox'])
        assert multi_scale_crop_results['img_shape'] in [(256, 256),
                                                         (204, 204)]

        # MultiScaleCrop with random crop.
        imgs = list(np.random.rand(2, 256, 341, 3))
        results = dict(imgs=imgs)
        config = dict(
            input_size=224,
            scales=(1, 0.8),
            random_crop=True,
            max_wh_scale_gap=0)
        multi_scale_crop = MultiScaleCrop(**config)
        multi_scale_crop_results = multi_scale_crop(results)
        assert assert_dict_has_keys(multi_scale_crop_results, target_keys)
        assert check_crop(imgs, multi_scale_crop_results['imgs'],
                          multi_scale_crop_results['crop_bbox'])
        assert (multi_scale_crop_results['img_shape'] in [(256, 256),
                                                          (204, 204)])

        assert repr(multi_scale_crop) == (
            f'{multi_scale_crop.__class__.__name__}'
            f'(input_size={(224, 224)}, scales={(1, 0.8)}, '
            f'max_wh_scale_gap={0}, random_crop={True}, '
            f'num_fixed_crops=5, lazy={False})')

    @staticmethod
    def test_center_crop():
        with pytest.raises(TypeError):
            # crop_size must be int or tuple of int
            CenterCrop(0.5)

        with pytest.raises(TypeError):
            # crop_size must be int or tuple of int
            CenterCrop('224')

        # center crop with crop_size 224
        # add kps in test_center_crop
        imgs = list(np.random.rand(2, 240, 320, 3))
        results = dict(imgs=imgs)
        kp = np.array([[160, 120], [160, 120]]).reshape([1, 1, 2, 2])
        results['keypoint'] = kp

        results['gt_bboxes'] = np.array([[0, 0, 320, 240]])
        results['proposals'] = np.array([[0, 0, 320, 240]])
        center_crop = CenterCrop(crop_size=224)
        center_crop_results = center_crop(results)
        target_keys = ['imgs', 'crop_bbox', 'img_shape', 'keypoint']
        assert assert_dict_has_keys(center_crop_results, target_keys)
        assert check_crop(imgs, center_crop_results['imgs'],
                          center_crop_results['crop_bbox'])
        assert np.all(
            center_crop_results['crop_bbox'] == np.array([48, 8, 272, 232]))
        assert center_crop_results['img_shape'] == (224, 224)
        assert np.all(center_crop_results['keypoint'] == 112)

        assert repr(center_crop) == (f'{center_crop.__class__.__name__}'
                                     f'(crop_size={(224, 224)}, lazy={False})')

    @staticmethod
    def test_three_crop():
        with pytest.raises(TypeError):
            # crop_size must be int or tuple of int
            ThreeCrop(0.5)

        with pytest.raises(TypeError):
            # crop_size must be int or tuple of int
            ThreeCrop('224')

        # three crop with crop_size 120
        imgs = list(np.random.rand(2, 240, 120, 3))
        results = dict(imgs=imgs)
        three_crop = ThreeCrop(crop_size=120)
        three_crop_results = three_crop(results)
        target_keys = ['imgs', 'crop_bbox', 'img_shape']
        assert assert_dict_has_keys(three_crop_results, target_keys)
        assert check_crop(imgs, three_crop_results['imgs'],
                          three_crop_results['crop_bbox'], 3)
        assert three_crop_results['img_shape'] == (120, 120)

        # three crop with crop_size 224
        imgs = list(np.random.rand(2, 224, 224, 3))
        results = dict(imgs=imgs)
        three_crop = ThreeCrop(crop_size=224)
        three_crop_results = three_crop(results)
        target_keys = ['imgs', 'crop_bbox', 'img_shape']
        assert assert_dict_has_keys(three_crop_results, target_keys)
        assert check_crop(imgs, three_crop_results['imgs'],
                          three_crop_results['crop_bbox'], 3)
        assert three_crop_results['img_shape'] == (224, 224)

        assert repr(three_crop) == (f'{three_crop.__class__.__name__}'
                                    f'(crop_size={(224, 224)})')

    @staticmethod
    def test_ten_crop():
        with pytest.raises(TypeError):
            # crop_size must be int or tuple of int
            TenCrop(0.5)

        with pytest.raises(TypeError):
            # crop_size must be int or tuple of int
            TenCrop('224')

        # ten crop with crop_size 256
        imgs = list(np.random.rand(2, 256, 256, 3))
        results = dict(imgs=imgs)
        ten_crop = TenCrop(crop_size=224)
        ten_crop_results = ten_crop(results)
        target_keys = ['imgs', 'crop_bbox', 'img_shape']
        assert assert_dict_has_keys(ten_crop_results, target_keys)
        assert check_crop(imgs, ten_crop_results['imgs'],
                          ten_crop_results['crop_bbox'], 10)
        assert ten_crop_results['img_shape'] == (224, 224)

        assert repr(ten_crop) == (f'{ten_crop.__class__.__name__}'
                                  f'(crop_size={(224, 224)})')
