# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
from mmcv.utils import assert_dict_has_keys
from numpy.testing import assert_array_almost_equal

from mmaction.datasets.pipelines import CenterCrop, Imgaug
from .base import check_flip


class TestAugumentations:

    @staticmethod
    def test_imgaug():

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
        assert_dict_has_keys(default_results, target_keys)
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
        assert assert_dict_has_keys(flip_results, target_keys)
        assert check_flip(imgs, flip_results['imgs'], 'horizontal')
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
        assert_dict_has_keys(crop_results, target_keys)
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
        assert_dict_has_keys(resize_results, target_keys)
        assert resize_results['img_shape'] == (32, 32)
        assert repr(imgaug_resize) == f'Imgaug(transforms={transforms})'
