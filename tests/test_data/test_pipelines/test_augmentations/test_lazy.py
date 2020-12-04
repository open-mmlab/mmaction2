import numpy as np
import pytest
from tests.test_data.test_pipelines.test_augmentations.test_base_aug import \
    TestAugumentations

from mmaction.datasets.pipelines import (CenterCrop, Fuse, MultiScaleCrop,
                                         RandomCrop, RandomResizedCrop)


class TestLazy(TestAugumentations):

    def test_random_crop_lazy(self):
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
        assert self.check_keys_contain(random_crop_result.keys(), target_keys)
        assert id(imgs) == id(random_crop_result['imgs'])
        random_crop_result_fuse = Fuse()(random_crop_result)
        assert 'lazy' not in random_crop_result_fuse
        assert self.check_crop(imgs, random_crop_result_fuse['imgs'],
                               results['crop_bbox'])
        h, w = random_crop_result_fuse['img_shape']
        assert h == w == 224

        # Test the case that no need for cropping
        imgs = list(np.random.rand(2, 224, 224, 3))
        results = dict(imgs=imgs)
        random_crop = RandomCrop(size=224, lazy=True)
        random_crop_result = random_crop(results)
        assert self.check_keys_contain(random_crop_result.keys(), target_keys)
        assert id(imgs) == id(random_crop_result['imgs'])
        random_crop_result_fuse = Fuse()(random_crop_result)
        assert 'lazy' not in random_crop_result_fuse
        assert self.check_crop(imgs, random_crop_result_fuse['imgs'],
                               results['crop_bbox'])
        h, w = random_crop_result_fuse['img_shape']
        assert h == w == 224

        # Test the one-side-equal case
        imgs = list(np.random.rand(2, 224, 225, 3))
        results = dict(imgs=imgs)
        random_crop = RandomCrop(size=224, lazy=True)
        random_crop_result = random_crop(results)
        assert self.check_keys_contain(random_crop_result.keys(), target_keys)
        assert id(imgs) == id(random_crop_result['imgs'])
        random_crop_result_fuse = Fuse()(random_crop_result)
        assert 'lazy' not in random_crop_result_fuse
        assert self.check_crop(imgs, random_crop_result_fuse['imgs'],
                               results['crop_bbox'])
        h, w = random_crop_result_fuse['img_shape']
        assert h == w == 224

        assert repr(random_crop) == (f'{random_crop.__class__.__name__}'
                                     f'(size={224}, lazy={True})')

    def test_center_crop_lazy(self):
        with pytest.raises(TypeError):
            # crop_size must be int or tuple of int
            CenterCrop(0.5)

        with pytest.raises(TypeError):
            # crop_size must be int or tuple of int
            CenterCrop('224')

        with pytest.raises(TypeError):
            # crop_size must be int or tuple of int
            CenterCrop([224, 224])

        # center crop with crop_size 224
        imgs = list(np.random.rand(2, 240, 320, 3))
        results = dict(imgs=imgs)
        center_crop = CenterCrop(crop_size=224, lazy=True)
        center_crop_results = center_crop(results)

        target_keys = ['imgs', 'crop_bbox', 'img_shape']
        assert self.check_keys_contain(center_crop_results.keys(), target_keys)
        center_crop_results_fuse = Fuse()(center_crop_results)
        assert self.check_crop(imgs, center_crop_results_fuse['imgs'],
                               center_crop_results['crop_bbox'])
        assert np.all(center_crop_results_fuse['crop_bbox'] == np.array(
            [48, 8, 272, 232]))
        assert center_crop_results_fuse['img_shape'] == (224, 224)

        assert repr(center_crop) == (f'{center_crop.__class__.__name__}'
                                     f'(crop_size={(224, 224)}, lazy={True})')

    def test_random_resized_crop_lazy(self):

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
        assert self.check_keys_contain(random_crop_result.keys(), target_keys)
        assert id(imgs) == id(random_crop_result['imgs'])
        random_crop_result_fuse = Fuse()(random_crop_result)
        assert self.check_crop(imgs, random_crop_result_fuse['imgs'],
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
        assert self.check_keys_contain(random_crop_result.keys(), target_keys)
        assert id(imgs) == id(random_crop_result['imgs'])
        random_crop_result_fuse = Fuse()(random_crop_result)
        assert self.check_crop(imgs, random_crop_result_fuse['imgs'],
                               results['crop_bbox'])
        h, w = random_crop_result['img_shape']
        assert h == w == 256

    def test_multi_scale_crop_lazy(self):
        with pytest.raises(TypeError):
            # input_size must be int or tuple of int
            MultiScaleCrop(0.5, lazy=True)

        with pytest.raises(TypeError):
            # input_size must be int or tuple of int
            MultiScaleCrop('224', lazy=True)

        with pytest.raises(TypeError):
            # input_size must be int or tuple of int
            MultiScaleCrop([224, 224], lazy=True)

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
        assert self.check_keys_contain(multi_scale_crop_result.keys(),
                                       target_keys)
        multi_scale_crop_result_fuse = Fuse()(multi_scale_crop_result)
        assert self.check_crop(imgs, multi_scale_crop_result_fuse['imgs'],
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
        assert self.check_keys_contain(multi_scale_crop_result.keys(),
                                       target_keys)
        multi_scale_crop_result_fuse = Fuse()(multi_scale_crop_result)
        assert self.check_crop(imgs, multi_scale_crop_result_fuse['imgs'],
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
        assert self.check_keys_contain(multi_scale_crop_result.keys(),
                                       target_keys)
        multi_scale_crop_result_fuse = Fuse()(multi_scale_crop_result)
        assert self.check_crop(imgs, multi_scale_crop_result_fuse['imgs'],
                               multi_scale_crop_result['crop_bbox'])
        assert (multi_scale_crop_result_fuse['img_shape'] in [(256, 256),
                                                              (204, 204)])

        assert repr(multi_scale_crop) == (
            f'{multi_scale_crop.__class__.__name__}'
            f'(input_size={(224, 224)}, scales={(1, 0.8)}, '
            f'max_wh_scale_gap={0}, random_crop={True}, '
            f'num_fixed_crops={5}, lazy={True})')
