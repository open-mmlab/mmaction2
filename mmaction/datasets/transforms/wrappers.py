# Copyright (c) OpenMMLab. All rights reserved.
import random

import mmengine
import numpy as np
from mmcv.transforms import BaseTransform, to_tensor
from mmengine.utils import digit_version

from mmaction.registry import TRANSFORMS


@TRANSFORMS.register_module()
class TorchVisionWrapper(BaseTransform):
    """Torchvision Augmentations, under torchvision.transforms.

    Args:
        op (str): The name of the torchvision transformation.
    """

    def __init__(self, op, **kwargs):
        try:
            import torchvision
            import torchvision.transforms as tv_trans
        except ImportError:
            raise RuntimeError('Install torchvision to use TorchvisionTrans')
        if digit_version(torchvision.__version__) < digit_version('0.8.0'):
            raise RuntimeError('The version of torchvision should be at least '
                               '0.8.0')

        trans = getattr(tv_trans, op, None)
        assert trans, f'Transform {op} not in torchvision'
        self.trans = trans(**kwargs)

    def transform(self, results):
        assert 'imgs' in results

        imgs = [x.transpose(2, 0, 1) for x in results['imgs']]
        imgs = to_tensor(np.stack(imgs))

        imgs = self.trans(imgs).data.numpy()
        imgs[imgs > 255] = 255
        imgs[imgs < 0] = 0
        imgs = imgs.astype(np.uint8)
        imgs = [x.transpose(1, 2, 0) for x in imgs]
        results['imgs'] = imgs
        return results


@TRANSFORMS.register_module()
class PytorchVideoWrapper(BaseTransform):
    """PytorchVideoTrans Augmentations, under pytorchvideo.transforms.

    Args:
        op (str): The name of the pytorchvideo transformation.
    """

    def __init__(self, op, **kwargs):
        try:
            import pytorchvideo.transforms as ptv_trans
            import torch
        except ImportError:
            raise RuntimeError('Install pytorchvideo to use PytorchVideoTrans')
        if digit_version(torch.__version__) < digit_version('1.8.0'):
            raise RuntimeError(
                'The version of PyTorch should be at least 1.8.0')

        trans = getattr(ptv_trans, op, None)
        assert trans, f'Transform {op} not in pytorchvideo'

        supported_pytorchvideo_trans = ('AugMix', 'RandAugment',
                                        'RandomResizedCrop', 'ShortSideScale',
                                        'RandomShortSideScale')
        assert op in supported_pytorchvideo_trans,\
            f'PytorchVideo Transform {op} is not supported in MMAction2'

        self.trans = trans(**kwargs)
        self.op = op

    def transform(self, results):
        assert 'imgs' in results

        assert 'gt_bboxes' not in results,\
            f'PytorchVideo {self.op} doesn\'t support bboxes yet.'
        assert 'proposals' not in results,\
            f'PytorchVideo {self.op} doesn\'t support bboxes yet.'

        if self.op in ('AugMix', 'RandAugment'):
            # list[ndarray(h, w, 3)] -> torch.tensor(t, c, h, w)
            imgs = [x.transpose(2, 0, 1) for x in results['imgs']]
            imgs = to_tensor(np.stack(imgs))
        else:
            # list[ndarray(h, w, 3)] -> torch.tensor(c, t, h, w)
            # uint8 -> float32
            imgs = to_tensor((np.stack(results['imgs']).transpose(3, 0, 1, 2) /
                              255.).astype(np.float32))

        imgs = self.trans(imgs).data.numpy()

        if self.op in ('AugMix', 'RandAugment'):
            imgs[imgs > 255] = 255
            imgs[imgs < 0] = 0
            imgs = imgs.astype(np.uint8)

            # torch.tensor(t, c, h, w) -> list[ndarray(h, w, 3)]
            imgs = [x.transpose(1, 2, 0) for x in imgs]
        else:
            # float32 -> uint8
            imgs = imgs * 255
            imgs[imgs > 255] = 255
            imgs[imgs < 0] = 0
            imgs = imgs.astype(np.uint8)

            # torch.tensor(c, t, h, w) -> list[ndarray(h, w, 3)]
            imgs = [x for x in imgs.transpose(1, 2, 3, 0)]

        results['imgs'] = imgs

        return results


@TRANSFORMS.register_module()
class ImgAug(BaseTransform):
    """Imgaug augmentation.

    Adds custom transformations from imgaug library.
    Please visit `https://imgaug.readthedocs.io/en/latest/index.html`
    to get more information. Two demo configs could be found in tsn and i3d
    config folder.

    It's better to use uint8 images as inputs since imgaug works best with
    numpy dtype uint8 and isn't well tested with other dtypes. It should be
    noted that not all of the augmenters have the same input and output dtype,
    which may cause unexpected results.

    Required keys are "imgs", "img_shape"(if "gt_bboxes" is not None) and
    "modality", added or modified keys are "imgs", "img_shape", "gt_bboxes"
    and "proposals".

    It is worth mentioning that `Imgaug` will NOT create custom keys like
    "interpolation", "crop_bbox", "flip_direction", etc. So when using
    `Imgaug` along with other mmaction2 pipelines, we should pay more attention
    to required keys.

    Two steps to use `Imgaug` pipeline:
    1. Create initialization parameter `transforms`. There are three ways
        to create `transforms`.
        1) string: only support `default` for now.
            e.g. `transforms='default'`
        2) list[dict]: create a list of augmenters by a list of dicts, each
            dict corresponds to one augmenter. Every dict MUST contain a key
            named `type`. `type` should be a string(iaa.Augmenter's name) or
            an iaa.Augmenter subclass.
            e.g. `transforms=[dict(type='Rotate', rotate=(-20, 20))]`
            e.g. `transforms=[dict(type=iaa.Rotate, rotate=(-20, 20))]`
        3) iaa.Augmenter: create an imgaug.Augmenter object.
            e.g. `transforms=iaa.Rotate(rotate=(-20, 20))`
    2. Add `Imgaug` in dataset pipeline. It is recommended to insert imgaug
        pipeline before `Normalize`. A demo pipeline is listed as follows.
        ```
        pipeline = [
            dict(
                type='SampleFrames',
                clip_len=1,
                frame_interval=1,
                num_clips=16,
            ),
            dict(type='RawFrameDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(
                type='MultiScaleCrop',
                input_size=224,
                scales=(1, 0.875, 0.75, 0.66),
                random_crop=False,
                max_wh_scale_gap=1,
                num_fixed_crops=13),
            dict(type='Resize', scale=(224, 224), keep_ratio=False),
            dict(type='Flip', flip_ratio=0.5),
            dict(type='Imgaug', transforms='default'),
            # dict(type='Imgaug', transforms=[
            #     dict(type='Rotate', rotate=(-20, 20))
            # ]),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='FormatShape', input_format='NCHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'label'])
        ]
        ```

    Args:
        transforms (str | list[dict] | :obj:`iaa.Augmenter`): Three different
            ways to create imgaug augmenter.
    """

    def __init__(self, transforms):
        import imgaug.augmenters as iaa

        if transforms == 'default':
            self.transforms = self.default_transforms()
        elif isinstance(transforms, list):
            assert all(isinstance(trans, dict) for trans in transforms)
            self.transforms = transforms
        elif isinstance(transforms, iaa.Augmenter):
            self.aug = self.transforms = transforms
        else:
            raise ValueError('transforms must be `default` or a list of dicts'
                             ' or iaa.Augmenter object')

        if not isinstance(transforms, iaa.Augmenter):
            self.aug = iaa.Sequential(
                [self.imgaug_builder(t) for t in self.transforms])

    @staticmethod
    def default_transforms():
        """Default transforms for imgaug.

        Implement RandAugment by imgaug.
        Please visit `https://arxiv.org/abs/1909.13719` for more information.

        Augmenters and hyper parameters are borrowed from the following repo:
        https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/autoaugment.py # noqa

        Miss one augmenter ``SolarizeAdd`` since imgaug doesn't support this.

        Returns:
            dict: The constructed RandAugment transforms.
        """
        # RandAugment hyper params
        num_augmenters = 2
        cur_magnitude, max_magnitude = 9, 10
        cur_level = 1.0 * cur_magnitude / max_magnitude

        return [
            dict(
                type='SomeOf',
                n=num_augmenters,
                children=[
                    dict(
                        type='ShearX',
                        shear=17.19 * cur_level * random.choice([-1, 1])),
                    dict(
                        type='ShearY',
                        shear=17.19 * cur_level * random.choice([-1, 1])),
                    dict(
                        type='TranslateX',
                        percent=.2 * cur_level * random.choice([-1, 1])),
                    dict(
                        type='TranslateY',
                        percent=.2 * cur_level * random.choice([-1, 1])),
                    dict(
                        type='Rotate',
                        rotate=30 * cur_level * random.choice([-1, 1])),
                    dict(type='Posterize', nb_bits=max(1, int(4 * cur_level))),
                    dict(type='Solarize', threshold=256 * cur_level),
                    dict(type='EnhanceColor', factor=1.8 * cur_level + .1),
                    dict(type='EnhanceContrast', factor=1.8 * cur_level + .1),
                    dict(
                        type='EnhanceBrightness', factor=1.8 * cur_level + .1),
                    dict(type='EnhanceSharpness', factor=1.8 * cur_level + .1),
                    dict(type='Autocontrast', cutoff=0),
                    dict(type='Equalize'),
                    dict(type='Invert', p=1.),
                    dict(
                        type='Cutout',
                        nb_iterations=1,
                        size=0.2 * cur_level,
                        squared=True)
                ])
        ]

    def imgaug_builder(self, cfg):
        """Import a module from imgaug.

        It follows the logic of :func:`build_from_cfg`. Use a dict object to
        create an iaa.Augmenter object.

        Args:
            cfg (dict): Config dict. It should at least contain the key "type".

        Returns:
            obj:`iaa.Augmenter`: The constructed imgaug augmenter.
        """
        import imgaug.augmenters as iaa

        assert isinstance(cfg, dict) and 'type' in cfg
        args = cfg.copy()

        obj_type = args.pop('type')
        if mmengine.is_str(obj_type):
            obj_cls = getattr(iaa, obj_type) if hasattr(iaa, obj_type) \
                else getattr(iaa.pillike, obj_type)
        elif issubclass(obj_type, iaa.Augmenter):
            obj_cls = obj_type
        else:
            raise TypeError(
                f'type must be a str or valid type, but got {type(obj_type)}')

        if 'children' in args:
            args['children'] = [
                self.imgaug_builder(child) for child in args['children']
            ]

        return obj_cls(**args)

    def __repr__(self):
        repr_str = self.__class__.__name__ + f'(transforms={self.aug})'
        return repr_str

    def transform(self, results):
        assert results['modality'] == 'RGB', 'Imgaug only support RGB images.'
        in_type = results['imgs'][0].dtype

        cur_aug = self.aug.to_deterministic()

        results['imgs'] = [
            cur_aug.augment_image(frame) for frame in results['imgs']
        ]
        img_h, img_w, _ = results['imgs'][0].shape

        out_type = results['imgs'][0].dtype
        assert in_type == out_type, \
            ('Imgaug input dtype and output dtype are not the same. ',
             f'Convert from {in_type} to {out_type}')

        if 'gt_bboxes' in results:
            from imgaug.augmentables import bbs
            bbox_list = [
                bbs.BoundingBox(
                    x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3])
                for bbox in results['gt_bboxes']
            ]
            bboxes = bbs.BoundingBoxesOnImage(
                bbox_list, shape=results['img_shape'])
            bbox_aug, *_ = cur_aug.augment_bounding_boxes([bboxes])
            results['gt_bboxes'] = [[
                max(bbox.x1, 0),
                max(bbox.y1, 0),
                min(bbox.x2, img_w),
                min(bbox.y2, img_h)
            ] for bbox in bbox_aug.items]
            if 'proposals' in results:
                bbox_list = [
                    bbs.BoundingBox(
                        x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3])
                    for bbox in results['proposals']
                ]
                bboxes = bbs.BoundingBoxesOnImage(
                    bbox_list, shape=results['img_shape'])
                bbox_aug, *_ = cur_aug.augment_bounding_boxes([bboxes])
                results['proposals'] = [[
                    max(bbox.x1, 0),
                    max(bbox.y1, 0),
                    min(bbox.x2, img_w),
                    min(bbox.y2, img_h)
                ] for bbox in bbox_aug.items]

        results['img_shape'] = (img_h, img_w)

        return results
