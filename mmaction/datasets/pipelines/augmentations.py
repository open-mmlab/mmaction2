import random
from collections.abc import Sequence

import mmcv
import numpy as np
from torch.nn.modules.utils import _pair

from ..registry import PIPELINES


@PIPELINES.register_module
class RandomCrop(object):
    """Vanilla square random crop that specifics the output size.

    Attributes:
        size (int): The output size of the images.
    """

    def __init__(self, size):
        if not isinstance(size, int):
            # size should be a int
            raise TypeError(f'size must be a int, but got {type(size)}')
        self.size = size

    def __call__(self, results):
        imgs = results['imgs']
        height, width = imgs.shape[1:3]
        # the cropping size should be less or equal to the origin size
        assert self.size <= height and self.size <= width

        if (height == self.size and width == self.size):
            results['crop_bbox'] = np.array([0, 0, self.size, self.size],
                                            dtype=np.int32)
            results['img_shape'] = results['imgs'].shape[1:3]

            return results

        y_offset = 0
        x_offset = 0
        if height > self.size:
            y_offset = int(np.random.randint(0, height - self.size))
        if width > self.size:
            x_offset = int(np.random.randint(0, width - self.size))

        results['crop_bbox'] = np.array(
            [x_offset, y_offset, x_offset + self.size, y_offset + self.size],
            dtype=np.int32)
        results['imgs'] = imgs[:, y_offset:y_offset + self.size,
                               x_offset:x_offset + self.size, :]

        results['img_shape'] = results['imgs'].shape[1:3]
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size})'
        return repr_str


@PIPELINES.register_module
class RandomResizedCrop(object):
    """Random crop that specifics the area and height-weight
        ratio range.

    Attributes:
        area_range (Tuple[float]): The candidate area scales range of
            output cropped images. Default: (0.08, 1.0).
        aspect_ratio_range (Tuple[float]): The candidate aspect ratio range of
            output cropped images. Default: (3 / 4, 4 / 3)
    """

    def __init__(self,
                 area_range=(0.08, 1.0),
                 aspect_ratio_range=(3 / 4, 4 / 3)):
        self.area_range = area_range
        self.aspect_ratio_range = aspect_ratio_range
        # area_range should be a tuple of float
        if not mmcv.is_tuple_of(self.area_range, float):
            raise TypeError(f'area_range must be a tuple of float, '
                            f'but got {type(area_range)}')
        # aspect_ratio_range should be a tuple of float
        if not mmcv.is_tuple_of(self.aspect_ratio_range, float):
            raise TypeError(f'aspect_ratio_range must be a tuple of float, '
                            f'but got {type(aspect_ratio_range)}')

    @staticmethod
    def get_crop_bbox(img, area_range, aspect_ratio_range, max_attempts=10):
        """Get a crop bbox given the area range and aspect ratio range.

        Args:
            area_range (Tuple[float]): The candidate area scales range of
                output cropped images. Default: (0.08, 1.0).
            aspect_ratio_range (Tuple[float]): The candidate aspect
                ratio range of output cropped images. Default: (3 / 4, 4 / 3).
                max_attempts (int): The maximum of attempts. Default: 10.
        Returns:
            A random crop bbox ggiven the area range and aspect ratio range.
        """
        # area_range must be greater than 0 and less than or equal to 1
        assert 0 < area_range[0] <= area_range[1] <= 1
        # aspect_ratio_range should not be negative
        assert 0 < aspect_ratio_range[0] <= aspect_ratio_range[1]

        area = img.shape[1] * img.shape[2]

        min_ar, max_ar = aspect_ratio_range
        aspect_ratios = np.exp(
            np.random.uniform(
                np.log(min_ar), np.log(max_ar), size=max_attempts))
        target_areas = np.random.uniform(*area_range, size=max_attempts) * area
        candidate_crop_w = np.round(np.sqrt(target_areas *
                                            aspect_ratios)).astype(np.int32)
        candidate_crop_h = np.round(np.sqrt(target_areas /
                                            aspect_ratios)).astype(np.int32)

        for i in range(max_attempts):
            crop_w = candidate_crop_w[i]
            crop_h = candidate_crop_h[i]
            if crop_h <= img.shape[1] and crop_w <= img.shape[2]:
                x_offset = random.randint(0, img.shape[2] - crop_w)
                y_offset = random.randint(0, img.shape[1] - crop_h)
                return x_offset, y_offset, x_offset + crop_w, y_offset + crop_h

        # Fallback
        crop_size = min(img.shape[1], img.shape[2])
        x_offset = (img.shape[2] - crop_size) // 2
        y_offset = (img.shape[1] - crop_size) // 2
        return x_offset, y_offset, x_offset + crop_size, y_offset + crop_size

    def __call__(self, results):
        imgs = results['imgs']

        left, top, right, bottom = self.get_crop_bbox(imgs, self.area_range,
                                                      self.aspect_ratio_range)

        results['crop_bbox'] = np.array([left, top, right, bottom],
                                        dtype=np.int32)
        results['imgs'] = imgs[:, top:bottom, left:right, :]

        results['img_shape'] = results['imgs'].shape[1:3]
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(area_range={self.area_range}, '
                     f'aspect_ratio_range={self.aspect_ratio_range})')
        return repr_str


@PIPELINES.register_module
class MultiScaleCrop(object):
    """Crop images with a list of randomly selected scales.

    Randomly select the w and h scales from a list of scales. Scale of 1 means
    the base size, which is the minimal of image weight and height. The scale
    level of w and h is controlled to be smaller than a certain value to
    prevent too large or small aspect ratio.
    Required keys are "imgs", added or modified keys are "imgs", "crop_bbox",
    "img_shape" and "scales".

    Attributes:
        input_size (int | tuple[int]): (w, h) of network input.
        scales (tuple[float]): Weight and height scales to be selected.
        max_wh_scale_gap (int): Maximum gap of w and h scale levels.
            Default: 1.
        random_crop (bool): If set to True, the cropping bbox will be randomly
            sampled, otherwise it will be sampler from fixed regions.
            Default: False.
        num_fixed_crops (int):
            If set to 5, the cropping bbox will keep 5 basic fixed regions:
                "upper left", "upper right", "lower left",
                 "lower right", "center".
            If set to 13, the cropping bbox will append another 8 fix regions:
                "center left", "center right", "lower center",
                "upper center", "upper left quarter", "upper right quarter",
                "lower left quarter", "lower right quarter".
            Default: 5.
    """

    def __init__(self,
                 input_size,
                 scales=(1, ),
                 max_wh_scale_gap=1,
                 random_crop=False,
                 num_fixed_crops=5):
        self.input_size = _pair(input_size)
        if not mmcv.is_tuple_of(self.input_size, int):
            raise TypeError(f'Input_size must be int or tuple of int, '
                            f'but got {type(input_size)}')

        if not isinstance(scales, tuple):
            raise TypeError(f'Scales must be tuple, but got {type(scales)}')

        if num_fixed_crops not in [5, 13]:
            raise ValueError(f'Num_fix_crops must be in {[5, 13]}, '
                             f'but got {num_fixed_crops}')

        self.scales = scales
        self.max_wh_scale_gap = max_wh_scale_gap
        self.random_crop = random_crop
        self.num_fixed_crops = num_fixed_crops

    def __call__(self, results):
        imgs = results['imgs']
        img_h, img_w = imgs.shape[1:3]

        base_size = min(img_h, img_w)

        crop_sizes = [int(base_size * s) for s in self.scales]

        candidate_sizes = []
        for i, h in enumerate(crop_sizes):
            for j, w in enumerate(crop_sizes):
                if abs(i - j) <= self.max_wh_scale_gap:
                    candidate_sizes.append([w, h])

        crop_size = random.choice(candidate_sizes)
        for i in range(2):
            if abs(crop_size[i] - self.input_size[i]) < 3:
                crop_size[i] = self.input_size[i]

        crop_w, crop_h = crop_size

        if self.random_crop:
            x_offset = random.randint(0, img_w - crop_w)
            y_offset = random.randint(0, img_h - crop_h)
        else:
            w_step = (img_w - crop_w) // 4
            h_step = (img_h - crop_h) // 4
            candidate_offsets = [
                (0, 0),  # upper left
                (4 * w_step, 0),  # upper right
                (0, 4 * h_step),  # lower left
                (4 * w_step, 4 * h_step),  # lower right
                (2 * w_step, 2 * h_step),  # center
            ]
            if self.num_fixed_crops == 13:
                extra_candidate_offsets = [
                    (0, 2 * h_step),  # center left
                    (4 * w_step, 2 * h_step),  # center right
                    (2 * w_step, 4 * h_step),  # lower center
                    (2 * w_step, 0 * h_step),  # upper center
                    (1 * w_step, 1 * h_step),  # upper left quarter
                    (3 * w_step, 1 * h_step),  # upper right quarter
                    (1 * w_step, 3 * h_step),  # lower left quarter
                    (3 * w_step, 3 * h_step)  # lower right quarter
                ]
                candidate_offsets.extend(extra_candidate_offsets)
            x_offset, y_offset = random.choice(candidate_offsets)

        results['crop_bbox'] = np.array(
            [x_offset, y_offset, x_offset + crop_w, y_offset + crop_h],
            dtype=np.int32)
        results['imgs'] = imgs[:, y_offset:y_offset + crop_h,
                               x_offset:x_offset + crop_w, :]

        results['img_shape'] = results['imgs'].shape[1:3]
        results['scales'] = self.scales
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(input_size={self.input_size}, scales={self.scales}, '
                     f'max_wh_scale_gap={self.max_wh_scale_gap}, '
                     f'random_crop={self.random_crop},'
                     f'num_fixed_crops={self.num_fixed_crops})')
        return repr_str


@PIPELINES.register_module
class Resize(object):
    """Resize images to a specific size.

    Required keys are "imgs", added or modified keys are "imgs", "img_shape",
    "keep_ratio", "scale_factor" and "resize_size".

    Attributes:
        scale (float | Tuple[int]): If keep_ratio is True, it serves as scaling
            factor or maximum size:
                If it is a float number, the image will be rescaled by this
                factor, else if it is a tuple of 2 integers, the image will
                be rescaled as large as possible within the scale.
            Otherwise, it serves as (w, h) of output size.
        keep_ratio (bool): If set to True, Images will be resized without
            changing the aspect ratio. Otherwise, it will resize images to a
            given size. Default: True.
        interpolation (str): Algorithm used for interpolation:
            "nearest" | "bilinear". Default: "bilinear".
    """

    def __init__(self, scale, keep_ratio=True, interpolation='bilinear'):
        if isinstance(scale, float):
            if scale <= 0:
                raise ValueError(f'Invalid scale {scale}, must be positive.')
        elif isinstance(scale, tuple):
            max_long_edge = max(scale)
            max_short_edge = min(scale)
            if max_short_edge == -1:
                # assign np.inf to long edge for rescaling short edge later.
                scale = (np.inf, max_long_edge)
        else:
            raise TypeError(
                f'Scale must be float or tuple of int, but got {type(scale)}')
        self.scale = scale
        self.keep_ratio = keep_ratio
        self.interpolation = interpolation

    def __call__(self, results):
        imgs = results['imgs']
        n, h, w, c = imgs.shape
        if self.keep_ratio:
            new_size, self.scale_factor = mmcv.rescale_size((w, h),
                                                            self.scale,
                                                            return_scale=True)
            out_w, out_h = new_size
        else:
            out_w, out_h = self.scale
            self.scale_factor = np.array(
                [out_w / w, out_h / h, out_w / w, out_h / h], dtype=np.float32)

        rimgs = np.empty((n, out_h, out_w, c), dtype=imgs.dtype)
        for i in range(n):
            mmcv.imresize(
                imgs[i], (out_w, out_h),
                out=rimgs[i],
                interpolation=self.interpolation)

        results['imgs'] = rimgs
        results['img_shape'] = results['imgs'].shape[1:3]
        results['keep_ratio'] = self.keep_ratio
        results['scale_factor'] = self.scale_factor

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(scale={self.scale}, keep_ratio={self.keep_ratio}, ' \
            f"interpolation='{self.interpolation}')"
        return repr_str


@PIPELINES.register_module
class Flip(object):
    """Flip the input images with a probability.

    Reverse the order of elements in the given imgs with a specific direction.
    The shape of the imgs is preserved, but the elements are reordered.
    Required keys are "imgs", added or modified keys are "imgs"
    and "flip_direction".

    Attributes:
        flip_ratio (float): Probability of implementing flip. Default: 0.5.
        direction (str): Flip imgs horizontally or vertically. Options are
            "horiziontal" | "vertival". Default: "horizontal".
    """
    _directions = ['horizontal', 'vertical']

    def __init__(self, flip_ratio=0.5, direction='horizontal'):
        if direction not in self._directions:
            raise ValueError(f'Direction {direction} is not supported. '
                             f'Currently support ones are {self._directions}')
        self.flip_ratio = flip_ratio
        self.direction = direction

    def __call__(self, results):
        if np.random.rand() < self.flip_ratio:
            flip = True
        else:
            flip = False

        if flip:
            for img in results['imgs']:
                mmcv.imflip_(img, self.direction)

        results['flip'] = flip
        results['flip_direction'] = self.direction

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(flip_ratio={self.flip_ratio}, ' \
            f"direction='{self.direction}')"
        return repr_str


@PIPELINES.register_module
class Normalize(object):
    """Normalize images with the given mean and std value.

    Required keys are "imgs", added or modified keys are "imgs"
    and "img_norm_cfg".

    Attributes:
        mean (Sequence[float]): Mean values of different channels.
        std (Sequence[float]): Std values of different channels.
        to_bgr (bool): Whether to convert channels from RGB to BGR.
            Default: False.
    """

    def __init__(self, mean, std, to_bgr=False):
        if not isinstance(mean, Sequence):
            raise TypeError(
                f'Mean must be list, tuple or np.ndarray, but got {type(mean)}'
            )

        if not isinstance(std, Sequence):
            raise TypeError(
                f'Std must be list, tuple or np.ndarray, but got {type(std)}')

        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_bgr = to_bgr

    def __call__(self, results):
        imgs = results['imgs'].astype(np.float32)

        for img in imgs:
            mmcv.imnormalize_(img, self.mean, self.std, self.to_bgr)

        results['imgs'] = imgs
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_bgr=self.to_bgr)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_bgr={self.to_bgr})'
        return repr_str


@PIPELINES.register_module
class CenterCrop(object):
    """Crop the center area from images.

    Required keys are "imgs", added or modified keys are "imgs", "crop_bbox"
    and "img_shape".

    Attributes:
        crop_size(int | tuple[int]): (w, h) of crop size.
    """

    def __init__(self, crop_size):
        self.crop_size = _pair(crop_size)
        if not mmcv.is_tuple_of(self.crop_size, int):
            raise TypeError(f'Crop_size must be int or tuple of int, '
                            f'but got {type(crop_size)}')

    def __call__(self, results):
        imgs = results['imgs']

        img_h, img_w = imgs.shape[1:3]
        crop_w, crop_h = self.crop_size

        left = (img_w - crop_w) // 2
        top = (img_h - crop_h) // 2
        right = left + crop_w
        bottom = top + crop_h
        results['crop_bbox'] = np.array([left, top, right, bottom])
        results['imgs'] = imgs[:, top:bottom, left:right, :]
        results['img_shape'] = results['imgs'].shape[1:3]

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(crop_size={self.crop_size})'
        return repr_str


@PIPELINES.register_module
class ThreeCrop(object):
    """Crop images into three crops.

    Crop the images equally into three crops with equal intervals along the
    shorter side.
    Required keys are "imgs", added or modified keys are "imgs", "crop_bbox"
    and "img_shape".

    Attributes:
        crop_size(int | tuple[int]): (w, h) of crop size.
    """

    def __init__(self, crop_size):
        self.crop_size = _pair(crop_size)
        if not mmcv.is_tuple_of(self.crop_size, int):
            raise TypeError(f'Crop_size must be int or tuple of int, '
                            f'but got {type(crop_size)}')

    def __call__(self, results):
        imgs = results['imgs']
        img_h, img_w = imgs.shape[1:3]
        crop_w, crop_h = self.crop_size
        assert crop_h == img_h or crop_w == img_w

        if crop_h == img_h:
            w_step = (img_w - crop_w) // 2
            offsets = [
                (0, 0),  # left
                (2 * w_step, 0),  # right
                (w_step, 0),  # middle
            ]
        elif crop_w == img_w:
            h_step = (img_h - crop_h) // 2
            offsets = [
                (0, 0),  # top
                (0, 2 * h_step),  # down
                (0, h_step),  # middle
            ]

        img_crops = []
        crop_bboxes = []
        for x_offset, y_offset in offsets:
            bbox = [x_offset, y_offset, x_offset + crop_w, y_offset + crop_h]
            crop = imgs[:, y_offset:y_offset + crop_h,
                        x_offset:x_offset + crop_w, :]
            img_crops.append(crop)
            crop_bboxes.extend([bbox for _ in range(imgs.shape[0])])

        crop_bboxes = np.array(crop_bboxes)
        imgs = np.concatenate(img_crops, axis=0)
        results['imgs'] = imgs
        results['crop_bbox'] = crop_bboxes
        results['img_shape'] = results['imgs'].shape[1:3]

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(crop_size={self.crop_size})'
        return repr_str


@PIPELINES.register_module
class TenCrop(object):
    """Crop the images into 10 crops (corner + center + flip).

    Crop the four corners and the center part of the image with the same
    given crop_size, and flip it horizontally.
    Required keys are "imgs", added or modified keys are "imgs", "crop_bbox"
    and "img_shape".

    Attributes:
        crop_size(int | tuple[int]): (w, h) of crop size.
    """

    def __init__(self, crop_size):
        self.crop_size = _pair(crop_size)
        if not mmcv.is_tuple_of(self.crop_size, int):
            raise TypeError(f'Crop_size must be int or tuple of int, '
                            f'but got {type(crop_size)}')

    def __call__(self, results):
        imgs = results['imgs']

        img_h, img_w = imgs.shape[1:3]
        crop_w, crop_h = self.crop_size

        w_step = (img_w - crop_w) // 4
        h_step = (img_h - crop_h) // 4

        offsets = [
            (0, 0),  # upper left
            (4 * w_step, 0),  # upper right
            (0, 4 * h_step),  # lower left
            (4 * w_step, 4 * h_step),  # lower right
            (2 * w_step, 2 * h_step),  # center
        ]

        img_crops = list()
        crop_bboxes = list()
        for x_offset, y_offsets in offsets:
            crop = imgs[:, y_offsets:y_offsets + crop_h,
                        x_offset:x_offset + crop_w, :]
            flip_crop = np.flip(crop, axis=2).copy()
            bbox = [x_offset, y_offsets, x_offset + crop_w, y_offsets + crop_h]
            img_crops.append(crop)
            img_crops.append(flip_crop)
            crop_bboxes.extend([bbox for _ in range(imgs.shape[0] * 2)])

        crop_bboxes = np.array(crop_bboxes)
        imgs = np.concatenate(img_crops, axis=0)
        results['imgs'] = imgs
        results['crop_bbox'] = crop_bboxes
        results['img_shape'] = results['imgs'].shape[1:3]

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(crop_size={self.crop_size})'
        return repr_str
