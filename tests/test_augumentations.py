import numpy as np
import pytest

from mmaction.datasets.pipelines import (CenterCrop, Flip, MultiScaleCrop,
                                         Normalize, RandomCrop,
                                         RandomResizedCrop, Resize, TenCrop,
                                         ThreeCrop)


class TestAugumentations(object):

    @staticmethod
    def assert_img_equal(img, ref_img, ratio_thr=0.999):
        """Check if img and ref_img are matched approximatively."""
        assert img.shape == ref_img.shape
        assert img.dtype == ref_img.dtype
        area = ref_img.shape[-1] * ref_img.shape[-2]
        diff = np.abs(img.astype('int32') - ref_img.astype('int32'))
        assert np.sum(diff <= 1) / float(area) > ratio_thr

    @staticmethod
    def check_keys_contain(result_keys, target_keys):
        """Check if all elements in target_keys is in result_keys."""
        return set(target_keys).issubset(set(result_keys))

    @staticmethod
    def check_crop(origin_imgs, result_imgs, result_bbox, num_crops=1):
        """Check if the result_bbox is in correspond to result_imgs."""

        def check_single_crop(origin_imgs, result_imgs, result_bbox):
            result_img_shape = (result_imgs.shape[1], result_imgs.shape[2])
            crop_w = result_bbox[2] - result_bbox[0]
            crop_h = result_bbox[3] - result_bbox[1]
            crop_shape = (crop_h, crop_w)
            if not crop_shape == result_img_shape:
                return False
            left, top, right, bottom = result_bbox
            return np.equal(origin_imgs[:, top:bottom, left:right, :],
                            result_imgs).all()

        if result_bbox.ndim == 1:
            return check_single_crop(origin_imgs, result_imgs, result_bbox)
        elif result_bbox.ndim == 2:
            num_batch = len(origin_imgs)
            for i, bbox in enumerate(result_bbox):
                if num_crops == 10:
                    if (i / num_batch) % 2 == 0:
                        flag = check_single_crop(
                            origin_imgs[i % num_batch:(i + 1) % num_batch],
                            result_imgs[i:(i + 1)], bbox)
                    else:
                        flag = check_single_crop(
                            origin_imgs[i % num_batch:(i + 1) % num_batch],
                            np.flip(result_imgs[i:(i + 1)], axis=2), bbox)
                else:
                    flag = check_single_crop(
                        origin_imgs[i % num_batch:(i + 1) % num_batch],
                        result_imgs[i:(i + 1)], bbox)
                if not flag:
                    return False
            return True
        else:
            # bbox has a wrong dimension
            return False

    @staticmethod
    def check_flip(origin_imgs, result_imgs, flip_type):
        """Check if the origin_imgs are flipped correctly into result_imgs
        in different flip_types"""
        n, h, w, c = origin_imgs.shape
        if flip_type == 'horizontal':
            # yapf: disable
            for i in range(n):
                for j in range(h):
                    for k in range(w):
                        for l in range(c):
                            if result_imgs[i, j, k, l] != origin_imgs[i, j, w - 1 - k, l]:  # noqa:E501
                                return False
            # yapf: enable
        else:
            # yapf: disable
            for i in range(n):
                for j in range(h):
                    for k in range(w):
                        for l in range(c):
                            if result_imgs[i, j, k, l] != origin_imgs[i, h - 1 - j, k, l]:  # noqa:E501
                                return False
            # yapf: enable
        return True

    def check_normalize(self, origin_imgs, result_imgs, norm_cfg):
        """Check if the origin_imgs are normalized correctly into result_imgs
         in a given norm_cfg."""
        target_imgs = result_imgs.copy()
        target_imgs *= norm_cfg['std']
        target_imgs += norm_cfg['mean']
        if norm_cfg['to_bgr']:
            target_imgs = target_imgs[..., ::-1].copy()
        self.assert_img_equal(origin_imgs, target_imgs)

    def test_random_crop(self):
        with pytest.raises(TypeError):
            # size must be an int
            RandomCrop(size=(112, 112))
        with pytest.raises(AssertionError):
            # "size > height" or "size > width" is not allowed
            imgs = np.random.rand(2, 224, 341, 3)
            results = dict(imgs=imgs)
            random_crop = RandomCrop(size=320)
            random_crop(results)

        target_keys = ['imgs', 'crop_bbox', 'img_shape']

        # General case
        imgs = np.random.rand(2, 224, 341, 3)
        results = dict(imgs=imgs)
        random_crop = RandomCrop(size=224)
        random_crop_result = random_crop(results)
        assert self.check_keys_contain(random_crop_result.keys(), target_keys)
        assert self.check_crop(imgs, random_crop_result['imgs'],
                               results['crop_bbox'])
        h, w = random_crop_result['img_shape']
        assert h == w == 224

        # Test the case that no need for cropping
        imgs = np.random.rand(2, 224, 224, 3)
        results = dict(imgs=imgs)
        random_crop = RandomCrop(size=224)
        random_crop_result = random_crop(results)
        assert self.check_keys_contain(random_crop_result.keys(), target_keys)
        assert self.check_crop(imgs, random_crop_result['imgs'],
                               results['crop_bbox'])
        h, w = random_crop_result['img_shape']
        assert h == w == 224

        # Test the one-side-equal case
        imgs = np.random.rand(2, 224, 225, 3)
        results = dict(imgs=imgs)
        random_crop = RandomCrop(size=224)
        random_crop_result = random_crop(results)
        assert self.check_keys_contain(random_crop_result.keys(), target_keys)
        assert self.check_crop(imgs, random_crop_result['imgs'],
                               results['crop_bbox'])
        h, w = random_crop_result['img_shape']
        assert h == w == 224

        assert repr(random_crop) == random_crop.__class__.__name__ +\
            f'(size=224)'

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
        imgs = np.random.rand(2, 256, 341, 3)
        results = dict(imgs=imgs)

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
        assert repr(random_crop) == random_crop.__class__.__name__ + \
            f'(area_range={(0.08, 1.0)}, aspect_ratio_range={(3 / 4, 4 / 3)})'

        random_crop = RandomResizedCrop(
            area_range=(0.9, 0.9), aspect_ratio_range=(10.0, 10.1))
        # Test fallback cases by very big area range
        imgs = np.random.rand(2, 256, 341, 3)
        results = dict(imgs=imgs)
        random_crop_result = random_crop(results)
        assert self.check_keys_contain(random_crop_result.keys(), target_keys)
        assert self.check_crop(imgs, random_crop_result['imgs'],
                               results['crop_bbox'])
        h, w = random_crop_result['img_shape']
        assert h == w == 256

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
        imgs = np.random.rand(2, 256, 341, 3)
        results = dict(imgs=imgs)
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
        imgs = np.random.rand(2, 256, 341, 3)
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
        imgs = np.random.rand(2, 256, 341, 3)
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

        assert repr(multi_scale_crop) == multi_scale_crop.__class__.__name__ +\
            '(input_size=(224, 224), scales=(1, 0.8), ' \
            'max_wh_scale_gap=0, random_crop=True,' \
            'num_fixed_crops=5)'

    def test_resize(self):
        with pytest.raises(ValueError):
            # scale must be positive
            Resize(-0.5)

        with pytest.raises(TypeError):
            # scale must be tuple of int
            Resize('224')

        target_keys = ['imgs', 'img_shape', 'keep_ratio', 'scale_factor']

        # scale with -1 to indicate np.inf
        imgs = np.random.rand(2, 240, 320, 3)
        results = dict(imgs=imgs)
        resize = Resize(scale=(-1, 256), keep_ratio=True)
        resize_results = resize(results)
        assert self.check_keys_contain(resize_results.keys(), target_keys)
        assert resize_results['scale_factor'] == 256 / 240
        assert resize_results['img_shape'] == (256, 341)

        # scale with a normal tuple (320, 320) to indicate np.inf
        imgs = np.random.rand(2, 240, 320, 3)
        results = dict(imgs=imgs)
        resize = Resize(scale=(320, 320), keep_ratio=False)
        resize_results = resize(results)
        assert self.check_keys_contain(resize_results.keys(), target_keys)
        assert np.all(resize_results['scale_factor'] == np.array(
            [1, 320 / 240, 1, 320 / 240], dtype=np.float32))
        assert resize_results['img_shape'] == (320, 320)

        # scale with a normal tuple (341, 256) to indicate np.inf
        imgs = np.random.rand(2, 240, 320, 3)
        results = dict(imgs=imgs)
        resize = Resize(scale=(341, 256), keep_ratio=False)
        resize_results = resize(results)
        assert self.check_keys_contain(resize_results.keys(), target_keys)
        assert np.all(resize_results['scale_factor'] == np.array(
            [341 / 320, 256 / 240, 341 / 320, 256 / 240], dtype=np.float32))
        assert resize_results['img_shape'] == (256, 341)

        assert repr(resize) == resize.__class__.__name__ +\
            '(scale=(341, 256), keep_ratio=False, ' \
            "interpolation='bilinear')"

    def test_flip(self):
        with pytest.raises(ValueError):
            # direction must be in ['horizontal', 'vertical']
            Flip(direction='vertically')

        target_keys = ['imgs', 'flip_direction']

        # do not flip imgs.
        imgs = np.random.rand(2, 64, 64, 3)
        results = dict(imgs=imgs.copy())
        flip = Flip(flip_ratio=0, direction='horizontal')
        flip_results = flip(results)
        assert self.check_keys_contain(flip_results.keys(), target_keys)
        assert np.equal(imgs, results['imgs']).all()
        assert id(flip_results['imgs']) == id(results['imgs'])
        assert flip_results['imgs'].shape == imgs.shape

        # always flip imgs horizontally.
        imgs = np.random.rand(2, 64, 64, 3)
        results = dict(imgs=imgs.copy())
        flip = Flip(flip_ratio=1, direction='horizontal')
        flip_results = flip(results)
        assert self.check_keys_contain(flip_results.keys(), target_keys)
        assert self.check_flip(imgs, flip_results['imgs'],
                               flip_results['flip_direction'])
        assert id(flip_results['imgs']) == id(results['imgs'])
        assert flip_results['imgs'].shape == imgs.shape

        # always flip imgs vertivally.
        imgs = np.random.rand(2, 64, 64, 3)
        results = dict(imgs=imgs.copy())
        flip = Flip(flip_ratio=1, direction='vertical')
        flip_results = flip(results)
        assert self.check_keys_contain(flip_results.keys(), target_keys)
        assert self.check_flip(imgs, flip_results['imgs'],
                               flip_results['flip_direction'])
        assert id(flip_results['imgs']) == id(results['imgs'])
        assert flip_results['imgs'].shape == imgs.shape

        assert repr(flip) == flip.__class__.__name__ +\
            "(flip_ratio=1, direction='vertical')"

    def test_normalize(self):
        with pytest.raises(TypeError):
            # mean must be list, tuple or np.ndarray
            Normalize(
                dict(mean=[123.675, 116.28, 103.53]), [58.395, 57.12, 57.375])

        with pytest.raises(TypeError):
            # std must be list, tuple or np.ndarray
            Normalize([123.675, 116.28, 103.53],
                      dict(std=[58.395, 57.12, 57.375]))

        target_keys = ['imgs', 'img_norm_cfg']

        # normalize imgs in RGB format
        imgs = np.random.rand(2, 240, 320, 3).astype(np.float32)
        results = dict(imgs=imgs)
        config = dict(
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_bgr=False)
        normalize = Normalize(**config)
        normalize_results = normalize(results)
        assert self.check_keys_contain(normalize_results.keys(), target_keys)
        self.check_normalize(imgs, normalize_results['imgs'],
                             normalize_results['img_norm_cfg'])

        # normalize imgs in BGR format
        imgs = np.random.rand(2, 240, 320, 3).astype(np.float32)
        results = dict(imgs=imgs)
        config = dict(
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_bgr=True)
        normalize = Normalize(**config)
        normalize_results = normalize(results)
        assert self.check_keys_contain(normalize_results.keys(), target_keys)
        self.check_normalize(imgs, normalize_results['imgs'],
                             normalize_results['img_norm_cfg'])

        assert normalize.__repr__() == normalize.__class__.__name__ +\
            f'(mean={np.array([123.675, 116.28, 103.53])}, ' \
            f'std={np.array([58.395, 57.12, 57.375])}, to_bgr=True)'

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
        imgs = np.random.rand(2, 240, 320, 3)
        results = dict(imgs=imgs)
        center_crop = CenterCrop(crop_size=224)
        center_crop_results = center_crop(results)
        target_keys = ['imgs', 'crop_bbox', 'img_shape']
        assert self.check_keys_contain(center_crop_results.keys(), target_keys)
        assert self.check_crop(imgs, center_crop_results['imgs'],
                               center_crop_results['crop_bbox'])
        assert np.all(
            center_crop_results['crop_bbox'] == np.array([48, 8, 272, 232]))
        assert center_crop_results['img_shape'] == (224, 224)

        assert repr(center_crop) == center_crop.__class__.__name__ + \
            '(crop_size=(224, 224))'

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
        imgs = np.random.rand(2, 240, 120, 3)
        results = dict(imgs=imgs)
        three_crop = ThreeCrop(crop_size=120)
        three_crop_results = three_crop(results)
        target_keys = ['imgs', 'crop_bbox', 'img_shape']
        assert self.check_keys_contain(three_crop_results.keys(), target_keys)
        assert self.check_crop(imgs, three_crop_results['imgs'],
                               three_crop_results['crop_bbox'], 3)
        assert three_crop_results['img_shape'] == (120, 120)

        # three crop with crop_size 224
        imgs = np.random.rand(2, 224, 224, 3)
        results = dict(imgs=imgs)
        three_crop = ThreeCrop(crop_size=224)
        three_crop_results = three_crop(results)
        target_keys = ['imgs', 'crop_bbox', 'img_shape']
        assert self.check_keys_contain(three_crop_results.keys(), target_keys)
        assert self.check_crop(imgs, three_crop_results['imgs'],
                               three_crop_results['crop_bbox'], 3)
        assert three_crop_results['img_shape'] == (224, 224)

        assert repr(three_crop) == three_crop.__class__.__name__ +\
            '(crop_size=(224, 224))'

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

        # ten crop with crop_size 224
        imgs = np.random.rand(2, 224, 224, 3)
        results = dict(imgs=imgs)
        ten_crop = TenCrop(crop_size=224)
        ten_crop_results = ten_crop(results)
        target_keys = ['imgs', 'crop_bbox', 'img_shape']
        assert self.check_keys_contain(ten_crop_results.keys(), target_keys)
        assert self.check_crop(imgs, ten_crop_results['imgs'],
                               ten_crop_results['crop_bbox'], 10)
        assert ten_crop_results['img_shape'] == (224, 224)

        assert repr(ten_crop) == ten_crop.__class__.__name__ +\
            '(crop_size=(224, 224))'
