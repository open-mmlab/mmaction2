import numpy as np
import pytest

from mmaction.datasets.pipelines import (CenterCrop, Flip, MultiScaleCrop,
                                         Normalize, Resize, TenCrop, ThreeCrop)


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
    def check_crop(result_img_shape, result_bbox):
        """Check if the result_bbox is in correspond to result_img_shape."""
        crop_w = result_bbox[2] - result_bbox[0]
        crop_h = result_bbox[3] - result_bbox[1]
        crop_shape = (crop_h, crop_w)
        return result_img_shape == crop_shape

    @staticmethod
    def check_flip(origin_imgs, result_imgs, flip_type):
        """Check if the origin_imgs are flipped correctly into result_imgs
        in different flip_types"""
        n, c, h, w = origin_imgs.shape
        if flip_type == 'horizontal':
            # yapf: disable
            for i in range(n):
                for j in range(c):
                    for k in range(h):
                        for l in range(w):
                            if result_imgs[i, j, k, l] != origin_imgs[i, j, k, w - 1 - l]:  # noqa:E501
                                return False
            # yapf: enable
        else:
            # yapf: disable
            for i in range(n):
                for j in range(c):
                    for k in range(h):
                        for l in range(w):
                            if result_imgs[i, j, k, l] != origin_imgs[i, j, h - 1 - k, l]:  # noqa:E501
                                return False
            # yapf: enable
        return True

    def check_normalize(self, origin_imgs, result_imgs, norm_cfg):
        """Check if the origin_imgs are normalized correctly into result_imgs
         in a given norm_cfg."""
        target_imgs = result_imgs.copy()
        target_imgs *= norm_cfg['std'][:, None, None]
        target_imgs += norm_cfg['mean'][:, None, None]
        if norm_cfg['to_bgr']:
            target_imgs = target_imgs[:, ::-1, ...].copy()
        self.assert_img_equal(origin_imgs, target_imgs)

    def test_multi_scale_crop(self):
        with pytest.raises(TypeError):
            MultiScaleCrop(0.5)

        with pytest.raises(TypeError):
            MultiScaleCrop('224')

        with pytest.raises(TypeError):
            MultiScaleCrop([224, 224])

        with pytest.raises(TypeError):
            MultiScaleCrop(
                224, scales=[
                    1,
                ])

        target_keys = ['imgs', 'crop_bbox', 'img_shape', 'scales']

        imgs = np.random.rand(2, 3, 256, 341)
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
        assert self.check_crop(multi_scale_crop_results['img_shape'],
                               multi_scale_crop_results['crop_bbox'])
        assert multi_scale_crop_results['img_shape'] in [(256, 256), (204, 204)
                                                         ]  # noqa: E501

        imgs = np.random.rand(2, 3, 256, 341)
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
        assert self.check_crop(multi_scale_crop_results['img_shape'],
                               multi_scale_crop_results['crop_bbox'])
        assert (multi_scale_crop_results['img_shape'] in [(256, 256),
                                                          (204, 204)])

        assert repr(multi_scale_crop) == multi_scale_crop.__class__.__name__ +\
            '(input_size={}, scales={}, max_wh_scale_gap={}, random_crop={})'.\
            format((224, 224), (1, 0.8), 0, True)

    def test_resize(self):
        with pytest.raises(ValueError):
            Resize(-0.5)

        with pytest.raises(TypeError):
            Resize('224')

        target_keys = ['imgs', 'img_shape', 'keep_ratio', 'scale_factor']

        imgs = np.random.rand(2, 3, 240, 320)
        results = dict(imgs=imgs)
        resize = Resize(scale=(-1, 256), keep_ratio=True)
        resize_results = resize(results)
        assert self.check_keys_contain(resize_results.keys(), target_keys)
        assert resize_results['scale_factor'] == 256 / 240
        assert resize_results['img_shape'] == (256, 341)

        imgs = np.random.rand(2, 3, 240, 320)
        results = dict(imgs=imgs)
        resize = Resize(scale=(320, 320), keep_ratio=False)
        resize_results = resize(results)
        assert self.check_keys_contain(resize_results.keys(), target_keys)
        assert np.all(resize_results['scale_factor'] == np.array(
            [1, 320 / 240, 1, 320 / 240], dtype=np.float32))
        assert resize_results['img_shape'] == (320, 320)

        imgs = np.random.rand(2, 3, 240, 320)
        results = dict(imgs=imgs)
        resize = Resize(scale=(341, 256), keep_ratio=False)
        resize_results = resize(results)
        assert self.check_keys_contain(resize_results.keys(), target_keys)
        assert np.all(resize_results['scale_factor'] == np.array(
            [341 / 320, 256 / 240, 341 / 320, 256 / 240], dtype=np.float32))
        assert resize_results['img_shape'] == (256, 341)

        assert repr(resize) == resize.__class__.__name__ +\
            '(scale={}, keep_ratio={}, interpolation={})'.format(
            (341, 256), False, 'bilinear')

    def test_flip(self):
        with pytest.raises(ValueError):
            Flip(direction='vertically')

        target_keys = ['imgs', 'flip_direction']

        imgs = np.random.rand(2, 3, 64, 64)
        results = dict(imgs=imgs)
        flip = Flip(flip_ratio=1, direction='horizontal')
        flip_results = flip(results)
        assert self.check_keys_contain(flip_results.keys(), target_keys)
        assert self.check_flip(imgs, flip_results['imgs'],
                               flip_results['flip_direction'])
        assert flip_results['imgs'].shape == imgs.shape

        imgs = np.random.rand(2, 3, 64, 64)
        results = dict(imgs=imgs)
        flip = Flip(flip_ratio=1, direction='vertical')
        flip_results = flip(results)
        assert self.check_keys_contain(flip_results.keys(), target_keys)
        assert self.check_flip(imgs, flip_results['imgs'],
                               flip_results['flip_direction'])
        assert flip_results['imgs'].shape == imgs.shape

        assert repr(flip) == flip.__class__.__name__ +\
            '(flip_ratio={}, direction={})'.format(1, 'vertical')

    def test_normalize(self):
        with pytest.raises(TypeError):
            Normalize(
                dict(mean=[123.675, 116.28, 103.53]), [58.395, 57.12, 57.375])

        with pytest.raises(TypeError):
            Normalize([123.675, 116.28, 103.53],
                      dict(std=[58.395, 57.12, 57.375]))

        target_keys = ['imgs', 'img_norm_cfg']

        imgs = np.random.rand(2, 3, 240, 320).astype(np.float32)
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

        imgs = np.random.rand(2, 3, 240, 320).astype(np.float32)
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
            '(mean={}, std={}, to_bgr={})'.format(
                np.array([123.675, 116.28, 103.53]),
                np.array([58.395, 57.12, 57.375]), True)

    def test_center_crop(self):
        with pytest.raises(TypeError):
            CenterCrop(0.5)

        with pytest.raises(TypeError):
            CenterCrop('224')

        with pytest.raises(TypeError):
            CenterCrop([224, 224])

        imgs = np.random.rand(2, 3, 240, 320)
        results = dict(imgs=imgs)
        center_crop = CenterCrop(crop_size=224)
        center_crop_results = center_crop(results)

        target_keys = ['imgs', 'crop_bbox', 'img_shape']
        assert self.check_keys_contain(center_crop_results.keys(), target_keys)
        assert self.check_crop(center_crop_results['img_shape'],
                               center_crop_results['crop_bbox'])
        assert np.all(
            center_crop_results['crop_bbox'] == np.array([48, 8, 272, 232]))
        assert center_crop_results['img_shape'] == (224, 224)

        assert repr(center_crop) == center_crop.__class__.__name__ + \
            '(crop_size={})'.format((224, 224))

    def test_three_crop(self):
        with pytest.raises(TypeError):
            ThreeCrop(0.5)

        with pytest.raises(TypeError):
            ThreeCrop('224')

        with pytest.raises(TypeError):
            ThreeCrop([224, 224])

        imgs = np.random.rand(2, 3, 224, 224)
        results = dict(imgs=imgs)
        three_crop = ThreeCrop(crop_size=224)
        three_crop_results = three_crop(results)

        target_keys = ['imgs', 'crop_bbox', 'img_shape']
        assert self.check_keys_contain(three_crop_results.keys(), target_keys)
        assert self.check_crop(three_crop_results['img_shape'],
                               three_crop_results['crop_bbox'][0])
        assert three_crop_results['img_shape'] == (224, 224)

        assert repr(three_crop) == three_crop.__class__.__name__ +\
            '(crop_size={})'.format((224, 224))

    def test_ten_crop(self):
        with pytest.raises(TypeError):
            TenCrop(0.5)

        with pytest.raises(TypeError):
            TenCrop('224')

        with pytest.raises(TypeError):
            TenCrop([224, 224])

        imgs = np.random.rand(2, 3, 224, 224)
        results = dict(imgs=imgs)
        ten_crop = TenCrop(crop_size=224)
        ten_crop_results = ten_crop(results)

        target_keys = ['imgs', 'crop_bbox', 'img_shape']
        assert self.check_keys_contain(ten_crop_results.keys(), target_keys)
        assert self.check_crop(ten_crop_results['img_shape'],
                               ten_crop_results['crop_bbox'][0])
        assert ten_crop_results['img_shape'] == (224, 224)

        assert ten_crop.__repr__() == ten_crop.__class__.__name__ +\
            '(crop_size={})'.format((224, 224))
