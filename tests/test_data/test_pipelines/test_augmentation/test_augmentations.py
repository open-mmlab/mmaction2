import copy

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from mmaction.datasets.pipelines import (CenterCrop, Imgaug, MultiGroupCrop,
                                         MultiScaleCrop, Normalize, RandomCrop,
                                         RandomRescale, RandomResizedCrop,
                                         RandomScale, Resize, TenCrop,
                                         ThreeCrop)


class TestAugumentations:

    @staticmethod
    def check_keys_contain(result_keys, target_keys):
        """Check if all elements in target_keys is in result_keys."""
        return set(target_keys).issubset(set(result_keys))

    @staticmethod
    def check_crop(origin_imgs, result_imgs, result_bbox, num_crops=1):
        """Check if the result_bbox is in correspond to result_imgs."""

        def check_single_crop(origin_imgs, result_imgs, result_bbox):
            result_img_shape = result_imgs[0].shape[:2]
            crop_w = result_bbox[2] - result_bbox[0]
            crop_h = result_bbox[3] - result_bbox[1]
            crop_shape = (crop_h, crop_w)
            if not crop_shape == result_img_shape:
                return False
            left, top, right, bottom = result_bbox
            return np.array_equal(
                np.array(origin_imgs)[:, top:bottom, left:right, :],
                np.array(result_imgs))

        if result_bbox.ndim == 1:
            return check_single_crop(origin_imgs, result_imgs, result_bbox)
        elif result_bbox.ndim == 2:
            num_batch = len(origin_imgs)
            for i, bbox in enumerate(result_bbox):
                if num_crops == 10:
                    if (i // num_batch) % 2 == 0:
                        flag = check_single_crop([origin_imgs[i % num_batch]],
                                                 [result_imgs[i]], bbox)
                    else:
                        flag = check_single_crop(
                            [origin_imgs[i % num_batch]],
                            [np.flip(result_imgs[i], axis=1)], bbox)
                else:
                    flag = check_single_crop([origin_imgs[i % num_batch]],
                                             [result_imgs[i]], bbox)
                if not flag:
                    return False
            return True
        else:
            # bbox has a wrong dimension
            return False

    @staticmethod
    def check_flip(origin_imgs, result_imgs, flip_type):
        """Check if the origin_imgs are flipped correctly into result_imgs in
        different flip_types."""
        n = len(origin_imgs)
        h, w, c = origin_imgs[0].shape
        if flip_type == 'horizontal':
            # yapf: disable
            for i in range(n):
                for j in range(h):
                    for k in range(w):
                        for channel in range(c):
                            if result_imgs[i][j, k, channel] != origin_imgs[i][j, w - 1 - k, channel]:  # noqa:E501
                                return False
            # yapf: enable
        else:
            # yapf: disable
            for i in range(n):
                for j in range(h):
                    for k in range(w):
                        for channel in range(c):
                            if result_imgs[i][j, k, channel] != origin_imgs[i][h - 1 - j, k, channel]:  # noqa:E501
                                return False
            # yapf: enable
        return True

    @staticmethod
    def check_normalize(origin_imgs, result_imgs, norm_cfg):
        """Check if the origin_imgs are normalized correctly into result_imgs
        in a given norm_cfg."""
        target_imgs = result_imgs.copy()
        target_imgs *= norm_cfg['std']
        target_imgs += norm_cfg['mean']
        if norm_cfg['to_bgr']:
            target_imgs = target_imgs[..., ::-1].copy()
        assert_array_almost_equal(origin_imgs, target_imgs, decimal=4)

    def test_random_crop(self):
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
        results['gt_bboxes'] = np.array([[0, 0, 340, 224]])
        results['proposals'] = None
        random_crop = RandomCrop(size=224)
        random_crop_result = random_crop(results)
        assert self.check_keys_contain(random_crop_result.keys(), target_keys)
        assert self.check_crop(imgs, random_crop_result['imgs'],
                               results['crop_bbox'])
        h, w = random_crop_result['img_shape']
        assert h == w == 224

        # Test the case that no need for cropping
        imgs = list(np.random.rand(2, 224, 224, 3))
        results = dict(imgs=imgs)
        random_crop = RandomCrop(size=224)
        random_crop_result = random_crop(results)
        assert self.check_keys_contain(random_crop_result.keys(), target_keys)
        assert self.check_crop(imgs, random_crop_result['imgs'],
                               results['crop_bbox'])
        h, w = random_crop_result['img_shape']
        assert h == w == 224

        # Test the one-side-equal case
        imgs = list(np.random.rand(2, 224, 225, 3))
        results = dict(imgs=imgs)
        random_crop = RandomCrop(size=224)
        random_crop_result = random_crop(results)
        assert self.check_keys_contain(random_crop_result.keys(), target_keys)
        assert self.check_crop(imgs, random_crop_result['imgs'],
                               results['crop_bbox'])
        h, w = random_crop_result['img_shape']
        assert h == w == 224

        assert repr(random_crop) == (f'{random_crop.__class__.__name__}'
                                     f'(size={224}, lazy={False})')

    def test_random_resized_crop(self):
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
        results['proposals'] = None

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
        assert self.check_keys_contain(random_crop_result.keys(), target_keys)
        assert self.check_crop(imgs, random_crop_result['imgs'],
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
        assert self.check_keys_contain(random_crop_result.keys(), target_keys)
        assert self.check_crop(imgs, random_crop_result['imgs'],
                               results['crop_bbox'])
        h, w = random_crop_result['img_shape']
        assert h == w == 256

    def test_random_rescale(self):
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

        assert self.check_keys_contain(random_rescale_result.keys(),
                                       target_keys)

        h, w = random_rescale_result['img_shape']

        # check rescale
        assert np.abs(h / 256 - w / 340) < eps
        assert 300 / 256 - eps <= h / 256 <= 400 / 256 + eps
        assert repr(random_rescale) == (f'{random_rescale.__class__.__name__}'
                                        f'(scale_range={(300, 400)}, '
                                        'interpolation=bilinear)')

    def test_multi_scale_crop(self):
        with pytest.raises(TypeError):
            # input_size must be int or tuple of int
            MultiScaleCrop(0.5)

        with pytest.raises(TypeError):
            # input_size must be int or tuple of int
            MultiScaleCrop('224')

        with pytest.raises(TypeError):
            # input_size must be int or tuple of int
            MultiScaleCrop([224, 224])

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
        results['proposals'] = None
        config = dict(
            input_size=224,
            scales=(1, 0.8),
            random_crop=False,
            max_wh_scale_gap=0)
        multi_scale_crop = MultiScaleCrop(**config)
        multi_scale_crop_results = multi_scale_crop(results)
        assert self.check_keys_contain(multi_scale_crop_results.keys(),
                                       target_keys)
        assert self.check_crop(imgs, multi_scale_crop_results['imgs'],
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
        assert self.check_keys_contain(multi_scale_crop_results.keys(),
                                       target_keys)
        assert self.check_crop(imgs, multi_scale_crop_results['imgs'],
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
        assert self.check_keys_contain(multi_scale_crop_results.keys(),
                                       target_keys)
        assert self.check_crop(imgs, multi_scale_crop_results['imgs'],
                               multi_scale_crop_results['crop_bbox'])
        assert (multi_scale_crop_results['img_shape'] in [(256, 256),
                                                          (204, 204)])

        assert repr(multi_scale_crop) == (
            f'{multi_scale_crop.__class__.__name__}'
            f'(input_size={(224, 224)}, scales={(1, 0.8)}, '
            f'max_wh_scale_gap={0}, random_crop={True}, '
            f'num_fixed_crops=5, lazy={False})')

    def test_resize(self):
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
        results = dict(imgs=imgs, modality='Flow')
        resize = Resize(scale=(160, 80), keep_ratio=False)
        resize_results = resize(results)
        assert self.check_keys_contain(resize_results.keys(), target_keys)
        assert np.all(resize_results['scale_factor'] == np.array(
            [.5, 1. / 3.], dtype=np.float32))
        assert resize_results['img_shape'] == (80, 160)

        # scale with -1 to indicate np.inf
        imgs = list(np.random.rand(2, 240, 320, 3))
        results = dict(imgs=imgs, modality='RGB')
        results['gt_bboxes'] = np.array([[0, 0, 320, 240]])
        results['proposals'] = None
        resize = Resize(scale=(-1, 256), keep_ratio=True)
        resize_results = resize(results)
        assert self.check_keys_contain(resize_results.keys(), target_keys)
        assert np.all(resize_results['scale_factor'] == np.array(
            [341 / 320, 256 / 240], dtype=np.float32))
        assert resize_results['img_shape'] == (256, 341)

        # scale with a normal tuple (320, 320) to indicate np.inf
        imgs = list(np.random.rand(2, 240, 320, 3))
        results = dict(imgs=imgs, modality='RGB')
        resize = Resize(scale=(320, 320), keep_ratio=False)
        resize_results = resize(results)
        assert self.check_keys_contain(resize_results.keys(), target_keys)
        assert np.all(resize_results['scale_factor'] == np.array(
            [1, 320 / 240], dtype=np.float32))
        assert resize_results['img_shape'] == (320, 320)

        # scale with a normal tuple (341, 256) to indicate np.inf
        imgs = list(np.random.rand(2, 240, 320, 3))
        results = dict(imgs=imgs, modality='RGB')
        resize = Resize(scale=(341, 256), keep_ratio=False)
        resize_results = resize(results)
        assert self.check_keys_contain(resize_results.keys(), target_keys)
        assert np.all(resize_results['scale_factor'] == np.array(
            [341 / 320, 256 / 240], dtype=np.float32))
        assert resize_results['img_shape'] == (256, 341)

        assert repr(resize) == (
            resize.__class__.__name__ +
            f'(scale={(341, 256)}, keep_ratio={False}, ' +
            f'interpolation=bilinear, lazy={False})')

    def test_normalize(self):
        with pytest.raises(TypeError):
            # mean must be list, tuple or np.ndarray
            Normalize(
                dict(mean=[123.675, 116.28, 103.53]), [58.395, 57.12, 57.375])

        with pytest.raises(TypeError):
            # std must be list, tuple or np.ndarray
            Normalize([123.675, 116.28, 103.53],
                      dict(std=[58.395, 57.12, 57.375]))

        target_keys = ['imgs', 'img_norm_cfg', 'modality']

        # normalize imgs in RGB format
        imgs = list(np.random.rand(2, 240, 320, 3).astype(np.float32))
        results = dict(imgs=imgs, modality='RGB')
        config = dict(
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_bgr=False)
        normalize = Normalize(**config)
        normalize_results = normalize(results)
        assert self.check_keys_contain(normalize_results.keys(), target_keys)
        self.check_normalize(imgs, normalize_results['imgs'],
                             normalize_results['img_norm_cfg'])

        # normalize flow imgs
        imgs = list(np.random.rand(4, 240, 320).astype(np.float32))
        results = dict(imgs=imgs, modality='Flow')
        config = dict(mean=[128, 128], std=[128, 128])
        normalize = Normalize(**config)
        normalize_results = normalize(results)
        assert self.check_keys_contain(normalize_results.keys(), target_keys)
        assert normalize_results['imgs'].shape == (2, 240, 320, 2)
        x_components = np.array(imgs[0::2])
        y_components = np.array(imgs[1::2])
        x_components = (x_components - config['mean'][0]) / config['std'][0]
        y_components = (y_components - config['mean'][1]) / config['std'][1]
        result_imgs = np.stack([x_components, y_components], axis=-1)
        assert np.all(np.isclose(result_imgs, normalize_results['imgs']))

        # normalize imgs in BGR format
        imgs = list(np.random.rand(2, 240, 320, 3).astype(np.float32))
        results = dict(imgs=imgs, modality='RGB')
        config = dict(
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_bgr=True)
        normalize = Normalize(**config)
        normalize_results = normalize(results)
        assert self.check_keys_contain(normalize_results.keys(), target_keys)
        self.check_normalize(imgs, normalize_results['imgs'],
                             normalize_results['img_norm_cfg'])

        assert normalize.__repr__() == (
            normalize.__class__.__name__ +
            f'(mean={np.array([123.675, 116.28, 103.53])}, ' +
            f'std={np.array([58.395, 57.12, 57.375])}, to_bgr={True}, '
            f'adjust_magnitude={False})')

    def test_center_crop(self):
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
        results['gt_bboxes'] = np.array([[0, 0, 320, 240]])
        results['proposals'] = None
        center_crop = CenterCrop(crop_size=224)
        center_crop_results = center_crop(results)
        target_keys = ['imgs', 'crop_bbox', 'img_shape']
        assert self.check_keys_contain(center_crop_results.keys(), target_keys)
        assert self.check_crop(imgs, center_crop_results['imgs'],
                               center_crop_results['crop_bbox'])
        assert np.all(
            center_crop_results['crop_bbox'] == np.array([48, 8, 272, 232]))
        assert center_crop_results['img_shape'] == (224, 224)

        assert repr(center_crop) == (f'{center_crop.__class__.__name__}'
                                     f'(crop_size={(224, 224)}, lazy={False})')

    def test_three_crop(self):
        with pytest.raises(TypeError):
            # crop_size must be int or tuple of int
            ThreeCrop(0.5)

        with pytest.raises(TypeError):
            # crop_size must be int or tuple of int
            ThreeCrop('224')

        with pytest.raises(TypeError):
            # crop_size must be int or tuple of int
            ThreeCrop([224, 224])

        # three crop with crop_size 120
        imgs = list(np.random.rand(2, 240, 120, 3))
        results = dict(imgs=imgs)
        three_crop = ThreeCrop(crop_size=120)
        three_crop_results = three_crop(results)
        target_keys = ['imgs', 'crop_bbox', 'img_shape']
        assert self.check_keys_contain(three_crop_results.keys(), target_keys)
        assert self.check_crop(imgs, three_crop_results['imgs'],
                               three_crop_results['crop_bbox'], 3)
        assert three_crop_results['img_shape'] == (120, 120)

        # three crop with crop_size 224
        imgs = list(np.random.rand(2, 224, 224, 3))
        results = dict(imgs=imgs)
        three_crop = ThreeCrop(crop_size=224)
        three_crop_results = three_crop(results)
        target_keys = ['imgs', 'crop_bbox', 'img_shape']
        assert self.check_keys_contain(three_crop_results.keys(), target_keys)
        assert self.check_crop(imgs, three_crop_results['imgs'],
                               three_crop_results['crop_bbox'], 3)
        assert three_crop_results['img_shape'] == (224, 224)

        assert repr(three_crop) == (f'{three_crop.__class__.__name__}'
                                    f'(crop_size={(224, 224)})')

    def test_ten_crop(self):
        with pytest.raises(TypeError):
            # crop_size must be int or tuple of int
            TenCrop(0.5)

        with pytest.raises(TypeError):
            # crop_size must be int or tuple of int
            TenCrop('224')

        with pytest.raises(TypeError):
            # crop_size must be int or tuple of int
            TenCrop([224, 224])

        # ten crop with crop_size 256
        imgs = list(np.random.rand(2, 256, 256, 3))
        results = dict(imgs=imgs)
        ten_crop = TenCrop(crop_size=224)
        ten_crop_results = ten_crop(results)
        target_keys = ['imgs', 'crop_bbox', 'img_shape']
        assert self.check_keys_contain(ten_crop_results.keys(), target_keys)
        assert self.check_crop(imgs, ten_crop_results['imgs'],
                               ten_crop_results['crop_bbox'], 10)
        assert ten_crop_results['img_shape'] == (224, 224)

        assert repr(ten_crop) == (f'{ten_crop.__class__.__name__}'
                                  f'(crop_size={(224, 224)})')

    def test_multi_group_crop(self):
        with pytest.raises(TypeError):
            # crop_size must be int or tuple of int
            MultiGroupCrop(0.5, 1)

        with pytest.raises(TypeError):
            # crop_size must be int or tuple of int
            MultiGroupCrop('224', 1)

        with pytest.raises(TypeError):
            # crop_size must be int or tuple of int
            MultiGroupCrop([224, 224], 1)

        with pytest.raises(TypeError):
            # groups must be int
            MultiGroupCrop(224, '1')

        with pytest.raises(ValueError):
            # groups must be positive
            MultiGroupCrop(224, 0)

        target_keys = ['imgs', 'crop_bbox', 'img_shape']

        # multi_group_crop with crop_size 224, groups 3
        imgs = list(np.random.rand(2, 256, 341, 3))
        results = dict(imgs=imgs)
        multi_group_crop = MultiGroupCrop(224, 3)
        multi_group_crop_result = multi_group_crop(results)
        assert self.check_keys_contain(multi_group_crop_result.keys(),
                                       target_keys)
        assert self.check_crop(imgs, multi_group_crop_result['imgs'],
                               multi_group_crop_result['crop_bbox'],
                               multi_group_crop.groups)
        assert multi_group_crop_result['img_shape'] == (224, 224)

        assert repr(multi_group_crop) == (
            f'{multi_group_crop.__class__.__name__}'
            f'(crop_size={(224, 224)}, groups={3})')

    def test_random_scale(self):
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

    def test_imgaug(self):

        with pytest.raises(ValueError):
            # transforms only support one string, 'default'
            Imgaug(transforms='test')

        with pytest.raises(ValueError):
            # transforms only support string or list of dicts
            # or iaa.Augmenter object
            Imgaug(transforms=dict(type='Rotate'))

        with pytest.raises(AssertionError):
            # each dict must have a `type` key
            Imgaug(transforms=[dict(rotate=(-30, 30))])

        with pytest.raises(AttributeError):
            # `type` must be available in imgaug
            Imgaug(transforms=[dict(type='BlaBla')])

        with pytest.raises(TypeError):
            # `type` must be str or iaa available type
            Imgaug(transforms=[dict(type=CenterCrop)])

        from imgaug import augmenters as iaa

        # check default configs
        target_keys = ['imgs', 'img_shape', 'modality']
        imgs = list(np.random.randint(0, 255, (1, 64, 64, 3)).astype(np.uint8))
        results = dict(imgs=imgs, modality='RGB')
        default_imgaug = Imgaug(transforms='default')
        default_results = default_imgaug(results)
        self.check_keys_contain(default_results.keys(), target_keys)
        assert default_results['img_shape'] == (64, 64)

        # check flip (both images and bboxes)
        target_keys = ['imgs', 'gt_bboxes', 'proposals', 'img_shape']
        imgs = list(np.random.rand(1, 64, 64, 3).astype(np.float32))
        results = dict(
            imgs=imgs,
            modality='RGB',
            proposals=np.array([[0, 0, 25, 35]]),
            img_shape=(64, 64),
            gt_bboxes=np.array([[0, 0, 25, 35]]))
        imgaug_flip = Imgaug(transforms=[dict(type='Fliplr')])
        flip_results = imgaug_flip(results)
        assert self.check_keys_contain(flip_results.keys(), target_keys)
        assert self.check_flip(imgs, flip_results['imgs'], 'horizontal')
        assert_array_almost_equal(flip_results['gt_bboxes'],
                                  np.array([[39, 0, 64, 35]]))
        assert_array_almost_equal(flip_results['proposals'],
                                  np.array([[39, 0, 64, 35]]))
        transforms = iaa.Sequential([iaa.Fliplr()])
        assert repr(imgaug_flip) == f'Imgaug(transforms={transforms})'

        # check crop (both images and bboxes)
        target_keys = ['crop_bbox', 'gt_bboxes', 'imgs', 'img_shape']
        imgs = list(np.random.rand(1, 122, 122, 3))
        results = dict(
            imgs=imgs,
            modality='RGB',
            img_shape=(122, 122),
            gt_bboxes=np.array([[1.5, 2.5, 110, 64]]))
        imgaug_center_crop = Imgaug(transforms=[
            dict(
                type=iaa.CropToFixedSize,
                width=100,
                height=100,
                position='center')
        ])
        crop_results = imgaug_center_crop(results)
        self.check_keys_contain(crop_results.keys(), target_keys)
        assert_array_almost_equal(crop_results['gt_bboxes'],
                                  np.array([[0., 0., 99., 53.]]))
        assert 'proposals' not in results
        transforms = iaa.Sequential(
            [iaa.CropToFixedSize(width=100, height=100, position='center')])
        assert repr(imgaug_center_crop) == f'Imgaug(transforms={transforms})'

        # check resize (images only)
        target_keys = ['imgs', 'img_shape']
        imgs = list(np.random.rand(1, 64, 64, 3))
        results = dict(imgs=imgs, modality='RGB')
        transforms = iaa.Resize(32)
        imgaug_resize = Imgaug(transforms=transforms)
        resize_results = imgaug_resize(results)
        self.check_keys_contain(resize_results.keys(), target_keys)
        assert resize_results['img_shape'] == (32, 32)
        assert repr(imgaug_resize) == f'Imgaug(transforms={transforms})'
