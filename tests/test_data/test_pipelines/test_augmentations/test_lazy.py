# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
from mmcv.utils import assert_dict_has_keys

from mmaction.datasets.pipelines import (CenterCrop, Flip, Fuse,
                                         MultiScaleCrop, RandomCrop,
                                         RandomResizedCrop, Resize)
from .base import check_crop, check_flip


class TestLazy:

    @staticmethod
    def test_init_lazy():
        from mmaction.datasets.pipelines.augmentations import \
            _init_lazy_if_proper  # noqa: E501
        with pytest.raises(AssertionError):
            # use lazy operation but "lazy" not in results
            result = dict(lazy=dict(), img_shape=[64, 64])
            _init_lazy_if_proper(result, False)

        lazy_keys = [
            'original_shape', 'crop_bbox', 'flip', 'flip_direction',
            'interpolation'
        ]

        # 'img_shape' not in results
        result = dict(imgs=list(np.random.randn(3, 64, 64, 3)))
        _init_lazy_if_proper(result, True)
        assert assert_dict_has_keys(result, ['imgs', 'lazy', 'img_shape'])
        assert assert_dict_has_keys(result['lazy'], lazy_keys)

        # 'img_shape' in results
        result = dict(img_shape=[64, 64])
        _init_lazy_if_proper(result, True)
        assert assert_dict_has_keys(result, ['lazy', 'img_shape'])
        assert assert_dict_has_keys(result['lazy'], lazy_keys)

        # do not use lazy operation
        result = dict(img_shape=[64, 64])
        _init_lazy_if_proper(result, False)
        assert assert_dict_has_keys(result, ['img_shape'])
        assert 'lazy' not in result

    @staticmethod
    def test_random_crop_lazy():
        with pytest.raises(TypeError):
            # size must be an int
            RandomCrop(size=(112, 112), lazy=True)
        with pytest.raises(AssertionError):
            # "size > height" or "size > width" is not allowed
            imgs = list(np.random.rand(2, 224, 341, 3))
            results = dict(imgs=imgs)
            random_crop = RandomCrop(size=320, lazy=True)
            random_crop(results)

        target_keys = ['imgs', 'crop_bbox', 'img_shape', 'lazy']

        # General case
        imgs = list(np.random.rand(2, 224, 341, 3))
        results = dict(imgs=imgs)
        random_crop = RandomCrop(size=224, lazy=True)
        random_crop_result = random_crop(results)
        assert assert_dict_has_keys(random_crop_result, target_keys)
        assert id(imgs) == id(random_crop_result['imgs'])
        random_crop_result_fuse = Fuse()(random_crop_result)
        assert 'lazy' not in random_crop_result_fuse
        assert check_crop(imgs, random_crop_result_fuse['imgs'],
                          results['crop_bbox'])
        h, w = random_crop_result_fuse['img_shape']
        assert h == w == 224

        # Test the case that no need for cropping
        imgs = list(np.random.rand(2, 224, 224, 3))
        results = dict(imgs=imgs)
        random_crop = RandomCrop(size=224, lazy=True)
        random_crop_result = random_crop(results)
        assert assert_dict_has_keys(random_crop_result, target_keys)
        assert id(imgs) == id(random_crop_result['imgs'])
        random_crop_result_fuse = Fuse()(random_crop_result)
        assert 'lazy' not in random_crop_result_fuse
        assert check_crop(imgs, random_crop_result_fuse['imgs'],
                          results['crop_bbox'])
        h, w = random_crop_result_fuse['img_shape']
        assert h == w == 224

        # Test the one-side-equal case
        imgs = list(np.random.rand(2, 224, 225, 3))
        results = dict(imgs=imgs)
        random_crop = RandomCrop(size=224, lazy=True)
        random_crop_result = random_crop(results)
        assert assert_dict_has_keys(random_crop_result, target_keys)
        assert id(imgs) == id(random_crop_result['imgs'])
        random_crop_result_fuse = Fuse()(random_crop_result)
        assert 'lazy' not in random_crop_result_fuse
        assert check_crop(imgs, random_crop_result_fuse['imgs'],
                          results['crop_bbox'])
        h, w = random_crop_result_fuse['img_shape']
        assert h == w == 224

        assert repr(random_crop) == (f'{random_crop.__class__.__name__}'
                                     f'(size={224}, lazy={True})')

    @staticmethod
    def test_random_resized_crop_lazy():

        target_keys = ['imgs', 'crop_bbox', 'img_shape', 'lazy']
        # There will be a slight difference because of rounding
        eps = 0.01
        imgs = list(np.random.rand(2, 256, 341, 3))
        results = dict(imgs=imgs)

        with pytest.raises(AssertionError):
            # area_range[0] > area_range[1], which is wrong
            random_crop = RandomResizedCrop(area_range=(0.9, 0.7), lazy=True)
            random_crop(results)
        with pytest.raises(AssertionError):
            # 0 > area_range[0] and area_range[1] > 1, which is wrong
            random_crop = RandomResizedCrop(
                aspect_ratio_range=(-0.1, 2.0), lazy=True)
            random_crop(results)

        random_crop = RandomResizedCrop(lazy=True)
        random_crop_result = random_crop(results)
        assert assert_dict_has_keys(random_crop_result, target_keys)
        assert id(imgs) == id(random_crop_result['imgs'])
        random_crop_result_fuse = Fuse()(random_crop_result)
        assert check_crop(imgs, random_crop_result_fuse['imgs'],
                          results['crop_bbox'])
        h, w = random_crop_result['img_shape']
        assert ((0.08 - eps <= h * w / 256 / 341)
                and (h * w / 256 / 341 <= 1 + eps))
        assert (3. / 4. - eps <= h / w) and (h / w - eps <= 4. / 3.)
        assert repr(random_crop) == (f'{random_crop.__class__.__name__}'
                                     f'(area_range={(0.08, 1.0)}, '
                                     f'aspect_ratio_range={(3 / 4, 4 / 3)}, '
                                     f'lazy={True})')

        random_crop = RandomResizedCrop(
            area_range=(0.9, 0.9), aspect_ratio_range=(10.0, 10.1), lazy=True)
        # Test fallback cases by very big area range
        imgs = np.random.rand(2, 256, 341, 3)
        results = dict(imgs=imgs)
        random_crop_result = random_crop(results)
        assert assert_dict_has_keys(random_crop_result, target_keys)
        assert id(imgs) == id(random_crop_result['imgs'])
        random_crop_result_fuse = Fuse()(random_crop_result)
        assert check_crop(imgs, random_crop_result_fuse['imgs'],
                          results['crop_bbox'])
        h, w = random_crop_result['img_shape']
        assert h == w == 256

    @staticmethod
    def test_multi_scale_crop_lazy():
        with pytest.raises(TypeError):
            # input_size must be int or tuple of int
            MultiScaleCrop(0.5, lazy=True)

        with pytest.raises(TypeError):
            # input_size must be int or tuple of int
            MultiScaleCrop('224', lazy=True)

        with pytest.raises(TypeError):
            # scales must be tuple.
            MultiScaleCrop(
                224, scales=[
                    1,
                ], lazy=True)

        with pytest.raises(ValueError):
            # num_fix_crops must be in [5, 13]
            MultiScaleCrop(224, num_fixed_crops=6, lazy=True)

        target_keys = ['imgs', 'crop_bbox', 'img_shape', 'scales']

        # MultiScaleCrop with normal crops.
        imgs = list(np.random.rand(2, 256, 341, 3))
        results = dict(imgs=imgs)
        config = dict(
            input_size=224,
            scales=(1, 0.8),
            random_crop=False,
            max_wh_scale_gap=0,
            lazy=True)
        multi_scale_crop = MultiScaleCrop(**config)
        multi_scale_crop_result = multi_scale_crop(results)
        assert id(imgs) == id(multi_scale_crop_result['imgs'])
        assert assert_dict_has_keys(multi_scale_crop_result, target_keys)
        multi_scale_crop_result_fuse = Fuse()(multi_scale_crop_result)
        assert check_crop(imgs, multi_scale_crop_result_fuse['imgs'],
                          multi_scale_crop_result['crop_bbox'])
        assert multi_scale_crop_result_fuse['img_shape'] in [(256, 256),
                                                             (204, 204)]

        # MultiScaleCrop with more fixed crops.
        imgs = list(np.random.rand(2, 256, 341, 3))
        results = dict(imgs=imgs)
        config = dict(
            input_size=224,
            scales=(1, 0.8),
            random_crop=False,
            max_wh_scale_gap=0,
            num_fixed_crops=13,
            lazy=True)
        multi_scale_crop = MultiScaleCrop(**config)
        multi_scale_crop_result = multi_scale_crop(results)
        assert id(imgs) == id(multi_scale_crop_result['imgs'])
        assert assert_dict_has_keys(multi_scale_crop_result, target_keys)
        multi_scale_crop_result_fuse = Fuse()(multi_scale_crop_result)
        assert check_crop(imgs, multi_scale_crop_result_fuse['imgs'],
                          multi_scale_crop_result['crop_bbox'])
        assert multi_scale_crop_result_fuse['img_shape'] in [(256, 256),
                                                             (204, 204)]

        # MultiScaleCrop with random crop.
        imgs = list(np.random.rand(2, 256, 341, 3))
        results = dict(imgs=imgs)
        config = dict(
            input_size=224,
            scales=(1, 0.8),
            random_crop=True,
            max_wh_scale_gap=0,
            lazy=True)
        multi_scale_crop = MultiScaleCrop(**config)
        multi_scale_crop_result = multi_scale_crop(results)
        assert id(imgs) == id(multi_scale_crop_result['imgs'])
        assert assert_dict_has_keys(multi_scale_crop_result, target_keys)
        multi_scale_crop_result_fuse = Fuse()(multi_scale_crop_result)
        assert check_crop(imgs, multi_scale_crop_result_fuse['imgs'],
                          multi_scale_crop_result['crop_bbox'])
        assert (multi_scale_crop_result_fuse['img_shape'] in [(256, 256),
                                                              (204, 204)])

        assert repr(multi_scale_crop) == (
            f'{multi_scale_crop.__class__.__name__}'
            f'(input_size={(224, 224)}, scales={(1, 0.8)}, '
            f'max_wh_scale_gap={0}, random_crop={True}, '
            f'num_fixed_crops={5}, lazy={True})')

    @staticmethod
    def test_resize_lazy():
        with pytest.raises(ValueError):
            # scale must be positive
            Resize(-0.5, lazy=True)

        with pytest.raises(TypeError):
            # scale must be tuple of int
            Resize('224', lazy=True)

        target_keys = [
            'imgs', 'img_shape', 'keep_ratio', 'scale_factor', 'modality'
        ]

        # scale with -1 to indicate np.inf
        imgs = list(np.random.rand(2, 240, 320, 3))
        results = dict(imgs=imgs, modality='RGB')
        resize = Resize(scale=(-1, 256), keep_ratio=True, lazy=True)
        resize_results = resize(results)
        assert id(imgs) == id(resize_results['imgs'])
        assert assert_dict_has_keys(resize_results, target_keys)
        resize_results_fuse = Fuse()(resize_results)
        assert np.all(resize_results_fuse['scale_factor'] == np.array(
            [341 / 320, 256 / 240], dtype=np.float32))
        assert resize_results_fuse['img_shape'] == (256, 341)

        # scale with a normal tuple (320, 320) to indicate np.inf
        imgs = list(np.random.rand(2, 240, 320, 3))
        results = dict(imgs=imgs, modality='RGB')
        resize = Resize(scale=(320, 320), keep_ratio=False, lazy=True)
        resize_results = resize(results)
        assert id(imgs) == id(resize_results['imgs'])
        assert assert_dict_has_keys(resize_results, target_keys)
        resize_results_fuse = Fuse()(resize_results)
        assert np.all(resize_results_fuse['scale_factor'] == np.array(
            [1, 320 / 240], dtype=np.float32))
        assert resize_results_fuse['img_shape'] == (320, 320)

        # scale with a normal tuple (341, 256) to indicate np.inf
        imgs = list(np.random.rand(2, 240, 320, 3))
        results = dict(imgs=imgs, modality='RGB')
        resize = Resize(scale=(341, 256), keep_ratio=False, lazy=True)
        resize_results = resize(results)
        assert id(imgs) == id(resize_results['imgs'])
        assert assert_dict_has_keys(resize_results, target_keys)
        resize_results_fuse = Fuse()(resize_results)
        assert np.all(resize_results_fuse['scale_factor'] == np.array(
            [341 / 320, 256 / 240], dtype=np.float32))
        assert resize_results_fuse['img_shape'] == (256, 341)

        assert repr(resize) == (f'{resize.__class__.__name__ }'
                                f'(scale={(341, 256)}, keep_ratio={False}, ' +
                                f'interpolation=bilinear, lazy={True})')

    @staticmethod
    def test_flip_lazy():
        with pytest.raises(ValueError):
            Flip(direction='vertically', lazy=True)

        target_keys = ['imgs', 'flip_direction', 'modality']

        # do not flip imgs.
        imgs = list(np.random.rand(2, 64, 64, 3))
        imgs_tmp = imgs.copy()
        results = dict(imgs=imgs_tmp, modality='RGB')
        flip = Flip(flip_ratio=0, direction='horizontal', lazy=True)
        flip_results = flip(results)
        assert id(imgs_tmp) == id(flip_results['imgs'])
        assert assert_dict_has_keys(flip_results, target_keys)
        flip_results_fuse = Fuse()(flip_results)
        assert np.equal(imgs, results['imgs']).all()
        assert id(flip_results['imgs']) == id(results['imgs'])
        assert flip_results_fuse['imgs'][0].shape == (64, 64, 3)

        # always flip imgs horizontally.
        imgs = list(np.random.rand(2, 64, 64, 3))
        imgs_tmp = imgs.copy()
        results = dict(imgs=imgs_tmp, modality='RGB')
        flip = Flip(flip_ratio=1, direction='horizontal', lazy=True)
        flip_results = flip(results)
        assert id(imgs_tmp) == id(flip_results['imgs'])
        assert assert_dict_has_keys(flip_results, target_keys)
        flip_results_fuse = Fuse()(flip_results)
        assert check_flip(imgs, flip_results['imgs'],
                          flip_results['flip_direction'])
        assert id(flip_results['imgs']) == id(results['imgs'])
        assert flip_results_fuse['imgs'][0].shape == (64, 64, 3)

        # always flip imgs vertivally.
        imgs = list(np.random.rand(2, 64, 64, 3))
        imgs_tmp = imgs.copy()
        results = dict(imgs=imgs_tmp, modality='RGB')
        flip = Flip(flip_ratio=1, direction='vertical', lazy=True)
        flip_results = flip(results)
        assert id(imgs_tmp) == id(flip_results['imgs'])
        assert assert_dict_has_keys(flip_results, target_keys)
        flip_results_fuse = Fuse()(flip_results)
        assert check_flip(imgs, flip_results['imgs'],
                          flip_results['flip_direction'])
        assert id(flip_results['imgs']) == id(results['imgs'])
        assert flip_results_fuse['imgs'][0].shape == (64, 64, 3)

        assert repr(flip) == (f'{flip.__class__.__name__}'
                              f'(flip_ratio={1}, direction=vertical, '
                              f'flip_label_map={None}, lazy={True})')

    @staticmethod
    def test_center_crop_lazy():
        with pytest.raises(TypeError):
            # crop_size must be int or tuple of int
            CenterCrop(0.5)

        with pytest.raises(TypeError):
            # crop_size must be int or tuple of int
            CenterCrop('224')

        # center crop with crop_size 224
        imgs = list(np.random.rand(2, 240, 320, 3))
        results = dict(imgs=imgs)
        center_crop = CenterCrop(crop_size=224, lazy=True)
        center_crop_results = center_crop(results)

        target_keys = ['imgs', 'crop_bbox', 'img_shape']
        assert assert_dict_has_keys(center_crop_results, target_keys)
        center_crop_results_fuse = Fuse()(center_crop_results)
        assert check_crop(imgs, center_crop_results_fuse['imgs'],
                          center_crop_results['crop_bbox'])
        assert np.all(center_crop_results_fuse['crop_bbox'] == np.array(
            [48, 8, 272, 232]))
        assert center_crop_results_fuse['img_shape'] == (224, 224)

        assert repr(center_crop) == (f'{center_crop.__class__.__name__}'
                                     f'(crop_size={(224, 224)}, lazy={True})')
