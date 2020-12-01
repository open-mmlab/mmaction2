import random
from collections.abc import Sequence

import mmcv
import numpy as np
from torch.nn.modules.utils import _pair

from ..registry import PIPELINES


def _init_lazy_if_proper(results, lazy):
    """Initialize lazy operation properly.

    Make sure that a lazy operation is properly initialized,
    and avoid a non-lazy operation accidentally getting mixed in.

    Required keys in results are "imgs" if "img_shape" not in results,
    otherwise, Required keys in results are "img_shape", add or modified keys
    are "img_shape", "lazy".
    Add or modified keys in "lazy" are "original_shape", "crop_bbox", "flip",
    "flip_direction", "interpolation".

    Args:
        results (dict): A dict stores data pipeline result.
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    if 'img_shape' not in results:
        results['img_shape'] = results['imgs'][0].shape[:2]
    if lazy:
        if 'lazy' not in results:
            img_h, img_w = results['img_shape']
            lazyop = dict()
            lazyop['original_shape'] = results['img_shape']
            lazyop['crop_bbox'] = np.array([0, 0, img_w, img_h],
                                           dtype=np.float32)
            lazyop['flip'] = False
            lazyop['flip_direction'] = None
            lazyop['interpolation'] = None
            results['lazy'] = lazyop
    else:
        assert 'lazy' not in results, 'Use Fuse after lazy operations'


@PIPELINES.register_module()
class Fuse:
    """Fuse lazy operations.

    Fusion order:
        crop -> resize -> flip

    Required keys are "imgs", "img_shape" and "lazy", added or modified keys
    are "imgs", "lazy".
    Required keys in "lazy" are "crop_bbox", "interpolation", "flip_direction".
    """

    def __call__(self, results):
        if 'lazy' not in results:
            raise ValueError('No lazy operation detected')
        lazyop = results['lazy']
        imgs = results['imgs']

        # crop
        left, top, right, bottom = lazyop['crop_bbox'].round().astype(int)
        imgs = [img[top:bottom, left:right] for img in imgs]

        # resize
        img_h, img_w = results['img_shape']
        if lazyop['interpolation'] is None:
            interpolation = 'bilinear'
        else:
            interpolation = lazyop['interpolation']
        imgs = [
            mmcv.imresize(img, (img_w, img_h), interpolation=interpolation)
            for img in imgs
        ]

        # flip
        if lazyop['flip']:
            for img in imgs:
                mmcv.imflip_(img, lazyop['flip_direction'])

        results['imgs'] = imgs
        del results['lazy']

        return results


@PIPELINES.register_module()
class RandomScale:
    """Resize images by a random scale.

    Required keys are "imgs", "img_shape", "modality", added or modified
    keys are "imgs", "img_shape", "keep_ratio", "scale_factor", "lazy",
    "scale", "resize_size". Required keys in "lazy" is None, added or
    modified key is "interpolation".

    Args:
        scales (tuple[int]): Tuple of scales to be chosen for resize.
        mode (str): Selection mode for choosing the scale. Options are "range"
            and "value". If set to "range", The short edge will be randomly
            chosen from the range of minimum and maximum on the shorter one
            in all tuples. Otherwise, the longer edge will be randomly chosen
            from the range of minimum and maximum on the longer one in all
            tuples. Default: 'range'.
    """

    def __init__(self, scales, mode='range', **kwargs):
        self.mode = mode
        if self.mode not in ['range', 'value']:
            raise ValueError(f"mode should be 'range' or 'value', "
                             f'but got {self.mode}')
        self.scales = scales
        self.kwargs = kwargs

    def select_scale(self, scales):
        num_scales = len(scales)
        if num_scales == 1:
            # specify a fixed scale
            scale = scales[0]
        elif num_scales == 2:
            if self.mode == 'range':
                scale_long = [max(s) for s in scales]
                scale_short = [min(s) for s in scales]
                long_edge = np.random.randint(
                    min(scale_long),
                    max(scale_long) + 1)
                short_edge = np.random.randint(
                    min(scale_short),
                    max(scale_short) + 1)
                scale = (long_edge, short_edge)
            elif self.mode == 'value':
                scale = random.choice(scales)
        else:
            if self.mode != 'value':
                raise ValueError("Only 'value' mode supports more than "
                                 '2 image scales')
            scale = random.choice(scales)

        return scale

    def __call__(self, results):
        scale = self.select_scale(self.scales)
        results['scale'] = scale
        resize = Resize(scale, **self.kwargs)
        results = resize(results)
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'scales={self.scales}, mode={self.mode})')
        return repr_str


@PIPELINES.register_module()
class EntityBoxRescale:
    """Rescale the entity box and proposals according to the image shape.

    Required keys are "img_shape", "scale_factor", "proposals",
    "ann.entity_boxes", added or modified keys are "ann.entity_boxes". If
    original "proposals" is not None, "proposals" and "scores" will be added or
    modified.
    """

    def __call__(self, results):
        img_h, img_w = results['img_shape']
        scale_factor = results['scale_factor']
        scale_factor = np.array([
            scale_factor[0], scale_factor[1], scale_factor[0], scale_factor[1]
        ])

        proposals = results['proposals']
        entity_boxes = results['ann']['entity_boxes']
        img_scale = np.array([img_w, img_h, img_w, img_h])
        entity_boxes = (entity_boxes * img_scale).astype(np.float32)
        results['ann']['entity_boxes'] = entity_boxes * scale_factor

        if proposals is not None:
            if proposals.shape[1] not in (4, 5):
                raise AssertionError('proposals shape should be in (n, 4) or '
                                     f'(n, 5), but got {proposals.shape}')
            if proposals.shape[1] == 5:
                scores = proposals[:, 4].astype(np.float32)
                proposals = proposals[:, :4]
            else:
                scores = None
            proposals = (proposals * img_scale).astype(np.float32)
            results['proposals'] = proposals * scale_factor
            results['scores'] = scores

        return results


@PIPELINES.register_module()
class EntityBoxCrop:
    """Crop the entity boxes and proposals according to the cropped images.

    Required keys are "proposals", "ann.entity_boxes", "crop_bbox", added or
    modified keys are "ann.entity_boxes". If original "proposals" is not None,
    "proposals" will be modified.
    """

    def __call__(self, results):
        proposals = results['proposals']
        entity_boxes = results['ann']['entity_boxes']
        crop_bboxes = results['crop_bbox']

        if crop_bboxes is None:
            return results
        x1, y1, _, _ = crop_bboxes

        assert entity_boxes.shape[-1] % 4 == 0
        entity_boxes_ = entity_boxes.copy()
        entity_boxes_[..., 0::2] = entity_boxes[..., 0::2] - x1
        entity_boxes_[..., 1::2] = entity_boxes[..., 1::2] - y1
        results['ann']['entity_boxes'] = entity_boxes_

        if proposals is not None:
            assert proposals.shape[-1] % 4 == 0
            proposals_ = proposals.copy()
            proposals_[..., 0::2] = proposals[..., 0::2] - x1
            proposals_[..., 1::2] = proposals[..., 1::2] - y1
            results['proposals'] = proposals_
        return results


@PIPELINES.register_module()
class EntityBoxFlip:
    """Flip the entity boxes and proposals with a probability.

    Reverse the order of elements in the given bounding boxes and proposals
    with a specific direction. The shape of them are preserved, but the
    elements are reordered.

    Required keys are "proposals", "img_shape", "ann.entity_boxes", added or
    modified keys are "flip", "flip_direction", "ann.entity_boxes". If
    "proposals" is not, it will also be modified.

    Args:
        flip_ratio (float): Probability of implementing flip. Default: 0.5.
        direction (str): Flip imgs horizontally or vertically. Options are
            "horizontal" | "vertical". Default: "horizontal".
    """

    _directions = ['horizontal', 'vertical']

    def __init__(self, flip_ratio=0.5, direction='horizontal'):
        if direction not in self._directions:
            raise ValueError(f'Direction {direction} is not supported. '
                             f'Currently support ones are {self._directions}')
        self.flip_ratio = flip_ratio
        self.direction = direction

    def __call__(self, results):
        flip = np.random.rand() < self.flip_ratio

        results['flip'] = flip
        results['flip_direction'] = self.direction

        proposals = results['proposals']
        entity_boxes = results['ann']['entity_boxes']
        img_h, img_w = results['img_shape']

        if flip:
            if self.direction == 'horizontal':
                assert entity_boxes.shape[-1] % 4 == 0
                entity_boxes_ = entity_boxes.copy()
                entity_boxes_[..., 0::4] = img_w - entity_boxes[..., 2::4] - 1
                entity_boxes_[..., 2::4] = img_w - entity_boxes[..., 0::4] - 1
                if proposals is not None:
                    assert proposals.shape[-1] % 4 == 0
                    proposals_ = proposals.copy()
                    proposals_[..., 0::4] = img_w - proposals[..., 2::4] - 1
                    proposals_[..., 2::4] = img_w - proposals[..., 0::4] - 1
                else:
                    proposals_ = None
            else:
                assert entity_boxes.shape[-1] % 4 == 0
                entity_boxes_ = entity_boxes.copy()
                entity_boxes_[..., 1::4] = img_h - entity_boxes[..., 3::4] - 1
                entity_boxes_[..., 3::4] = img_h - entity_boxes[..., 1::4] - 1
                if proposals is not None:
                    assert proposals.shape[-1] % 4 == 0
                    proposals_ = proposals.copy()
                    proposals_[..., 1::4] = img_h - proposals[..., 3::4] - 1
                    proposals_[..., 3::4] = img_h - proposals[..., 1::4] - 1
                else:
                    proposals_ = None

            results['proposals'] = proposals_
            results['ann']['entity_boxes'] = entity_boxes_
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'flip_ratio={self.flip_ratio}, '
                    f'direction={self.direction})')
        return repr_str


@PIPELINES.register_module()
class EntityBoxClip:
    """Clip (limit) the values in the entity boxes and proposals.

    Required keys are "img_shape", "proposals" and "ann.entity_boxes", added or
    modified keys are "ann.entity_boxes". If "proposals" is None, it will also
    be modified.
    """

    def __call__(self, results):
        proposals = results['proposals']
        entity_boxes = results['ann']['entity_boxes']
        img_h, img_w = results['img_shape']

        entity_boxes[:, 0::2] = np.clip(entity_boxes[:, 0::2], 0, img_w - 1)
        entity_boxes[:, 1::2] = np.clip(entity_boxes[:, 1::2], 0, img_h - 1)
        if proposals is not None:
            proposals[:, 0::2] = np.clip(proposals[:, 0::2], 0, img_w - 1)
            proposals[:, 1::2] = np.clip(proposals[:, 1::2], 0, img_h - 1)

        results['ann']['entity_boxes'] = entity_boxes
        results['proposals'] = proposals
        return results


@PIPELINES.register_module()
class EntityBoxPad:
    """Pad entity boxes and proposals with zeros.

    Required keys are "proposals" and "ann.entity_boxes", added or modified
    keys are "ann.entity_boxes". If "proposals" is not None, it is also
    modified.

    Args:
        max_num_gts (int | None): maximum of ground truth proposals.
            Default: None.
    """

    def __init__(self, max_num_gts=None):
        self.max_num_gts = max_num_gts

    def __call__(self, results):
        if self.max_num_gts is None:
            return results

        proposals = results['proposals']
        entity_boxes = results['ann']['entity_boxes']
        num_gts = entity_boxes.shape[0]

        padded_entity_boxes = np.zeros((self.max_num_gts, 4), dtype=np.float32)
        padded_entity_boxes[:num_gts, :] = entity_boxes
        if proposals is not None:
            padded_proposals = np.zeros((self.max_num_gts, 4),
                                        dtype=np.float32)
            padded_proposals[:num_gts, :] = proposals
        else:
            padded_proposals = None

        results['proposals'] = padded_proposals
        results['ann']['entity_boxes'] = padded_entity_boxes
        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}(max_num_gts={self.max_num_gts})'
        return repr_str


@PIPELINES.register_module()
class RandomCrop:
    """Vanilla square random crop that specifics the output size.

    Required keys in results are "imgs" and "img_shape", added or
    modified keys are "imgs", "lazy"; Required keys in "lazy" are "flip",
    "crop_bbox", added or modified key is "crop_bbox".

    Args:
        size (int): The output size of the images.
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    def __init__(self, size, lazy=False):
        if not isinstance(size, int):
            raise TypeError(f'Size must be an int, but got {type(size)}')
        self.size = size
        self.lazy = lazy

    def __call__(self, results):
        """Performs the RandomCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        _init_lazy_if_proper(results, self.lazy)

        img_h, img_w = results['img_shape']
        assert self.size <= img_h and self.size <= img_w

        y_offset = 0
        x_offset = 0
        if img_h > self.size:
            y_offset = int(np.random.randint(0, img_h - self.size))
        if img_w > self.size:
            x_offset = int(np.random.randint(0, img_w - self.size))

        new_h, new_w = self.size, self.size

        results['crop_bbox'] = np.array(
            [x_offset, y_offset, x_offset + new_w, y_offset + new_h])
        results['img_shape'] = (new_h, new_w)

        if not self.lazy:
            results['imgs'] = [
                img[y_offset:y_offset + new_h, x_offset:x_offset + new_w]
                for img in results['imgs']
            ]
        else:
            lazyop = results['lazy']
            if lazyop['flip']:
                raise NotImplementedError('Put Flip at last for now')

            # record crop_bbox in lazyop dict to ensure only crop once in Fuse
            lazy_left, lazy_top, lazy_right, lazy_bottom = lazyop['crop_bbox']
            left = x_offset * (lazy_right - lazy_left) / img_w
            right = (x_offset + new_w) * (lazy_right - lazy_left) / img_w
            top = y_offset * (lazy_bottom - lazy_top) / img_h
            bottom = (y_offset + new_h) * (lazy_bottom - lazy_top) / img_h
            lazyop['crop_bbox'] = np.array([(lazy_left + left),
                                            (lazy_top + top),
                                            (lazy_left + right),
                                            (lazy_top + bottom)],
                                           dtype=np.float32)

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}(size={self.size}, '
                    f'lazy={self.lazy})')
        return repr_str


@PIPELINES.register_module()
class RandomResizedCrop:
    """Random crop that specifics the area and height-weight ratio range.

    Required keys in results are "imgs", "img_shape", "crop_bbox" and "lazy",
    added or modified keys are "imgs", "crop_bbox" and "lazy"; Required keys
    in "lazy" are "flip", "crop_bbox", added or modified key is "crop_bbox".

    Args:
        area_range (Tuple[float]): The candidate area scales range of
            output cropped images. Default: (0.08, 1.0).
        aspect_ratio_range (Tuple[float]): The candidate aspect ratio range of
            output cropped images. Default: (3 / 4, 4 / 3).
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    def __init__(self,
                 area_range=(0.08, 1.0),
                 aspect_ratio_range=(3 / 4, 4 / 3),
                 lazy=False):
        self.area_range = area_range
        self.aspect_ratio_range = aspect_ratio_range
        self.lazy = lazy
        if not mmcv.is_tuple_of(self.area_range, float):
            raise TypeError(f'Area_range must be a tuple of float, '
                            f'but got {type(area_range)}')
        if not mmcv.is_tuple_of(self.aspect_ratio_range, float):
            raise TypeError(f'Aspect_ratio_range must be a tuple of float, '
                            f'but got {type(aspect_ratio_range)}')

    @staticmethod
    def get_crop_bbox(img_shape,
                      area_range,
                      aspect_ratio_range,
                      max_attempts=10):
        """Get a crop bbox given the area range and aspect ratio range.

        Args:
            img_shape (Tuple[int]): Image shape
            area_range (Tuple[float]): The candidate area scales range of
                output cropped images. Default: (0.08, 1.0).
            aspect_ratio_range (Tuple[float]): The candidate aspect
                ratio range of output cropped images. Default: (3 / 4, 4 / 3).
                max_attempts (int): The maximum of attempts. Default: 10.
            max_attempts (int): Max attempts times to generate random candidate
                bounding box. If it doesn't qualified one, the center bounding
                box will be used.
        Returns:
            (list[int]) A random crop bbox within the area range and aspect
            ratio range.
        """
        assert 0 < area_range[0] <= area_range[1] <= 1
        assert 0 < aspect_ratio_range[0] <= aspect_ratio_range[1]

        img_h, img_w = img_shape
        area = img_h * img_w

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
            if crop_h <= img_h and crop_w <= img_w:
                x_offset = random.randint(0, img_w - crop_w)
                y_offset = random.randint(0, img_h - crop_h)
                return x_offset, y_offset, x_offset + crop_w, y_offset + crop_h

        # Fallback
        crop_size = min(img_h, img_w)
        x_offset = (img_w - crop_size) // 2
        y_offset = (img_h - crop_size) // 2
        return x_offset, y_offset, x_offset + crop_size, y_offset + crop_size

    def __call__(self, results):
        """Performs the RandomResizeCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        _init_lazy_if_proper(results, self.lazy)

        img_h, img_w = results['img_shape']

        left, top, right, bottom = self.get_crop_bbox(
            (img_h, img_w), self.area_range, self.aspect_ratio_range)
        new_h, new_w = bottom - top, right - left

        results['crop_bbox'] = np.array([left, top, right, bottom])
        results['img_shape'] = (new_h, new_w)

        if not self.lazy:
            results['imgs'] = [
                img[top:bottom, left:right] for img in results['imgs']
            ]
        else:
            lazyop = results['lazy']
            if lazyop['flip']:
                raise NotImplementedError('Put Flip at last for now')

            # record crop_bbox in lazyop dict to ensure only crop once in Fuse
            lazy_left, lazy_top, lazy_right, lazy_bottom = lazyop['crop_bbox']
            left = left * (lazy_right - lazy_left) / img_w
            right = right * (lazy_right - lazy_left) / img_w
            top = top * (lazy_bottom - lazy_top) / img_h
            bottom = bottom * (lazy_bottom - lazy_top) / img_h
            lazyop['crop_bbox'] = np.array([(lazy_left + left),
                                            (lazy_top + top),
                                            (lazy_left + right),
                                            (lazy_top + bottom)],
                                           dtype=np.float32)

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'area_range={self.area_range}, '
                    f'aspect_ratio_range={self.aspect_ratio_range}, '
                    f'lazy={self.lazy})')
        return repr_str


@PIPELINES.register_module()
class MultiScaleCrop:
    """Crop images with a list of randomly selected scales.

    Randomly select the w and h scales from a list of scales. Scale of 1 means
    the base size, which is the minimal of image weight and height. The scale
    level of w and h is controlled to be smaller than a certain value to
    prevent too large or small aspect ratio.
    Required keys are "imgs", "img_shape", added or modified keys are "imgs",
    "crop_bbox", "img_shape", "lazy" and "scales". Required keys in "lazy" are
    "crop_bbox", added or modified key is "crop_bbox".

    Args:
        input_size (int | tuple[int]): (w, h) of network input.
        scales (tuple[float]): Weight and height scales to be selected.
        max_wh_scale_gap (int): Maximum gap of w and h scale levels.
            Default: 1.
        random_crop (bool): If set to True, the cropping bbox will be randomly
            sampled, otherwise it will be sampler from fixed regions.
            Default: False.
        num_fixed_crops (int): If set to 5, the cropping bbox will keep 5
            basic fixed regions: "upper left", "upper right", "lower left",
            "lower right", "center". If set to 13, the cropping bbox will
            append another 8 fix regions: "center left", "center right",
            "lower center", "upper center", "upper left quarter",
            "upper right quarter", "lower left quarter", "lower right quarter".
            Default: 5.
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    def __init__(self,
                 input_size,
                 scales=(1, ),
                 max_wh_scale_gap=1,
                 random_crop=False,
                 num_fixed_crops=5,
                 lazy=False):
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
        self.lazy = lazy

    def __call__(self, results):
        """Performs the MultiScaleCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        _init_lazy_if_proper(results, self.lazy)

        img_h, img_w = results['img_shape']
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

        new_h, new_w = crop_h, crop_w

        results['crop_bbox'] = np.array(
            [x_offset, y_offset, x_offset + new_w, y_offset + new_h])
        results['img_shape'] = (new_h, new_w)
        results['scales'] = self.scales

        if not self.lazy:
            results['imgs'] = [
                img[y_offset:y_offset + new_h, x_offset:x_offset + new_w]
                for img in results['imgs']
            ]
        else:
            lazyop = results['lazy']
            if lazyop['flip']:
                raise NotImplementedError('Put Flip at last for now')

            # record crop_bbox in lazyop dict to ensure only crop once in Fuse
            lazy_left, lazy_top, lazy_right, lazy_bottom = lazyop['crop_bbox']
            left = x_offset * (lazy_right - lazy_left) / img_w
            right = (x_offset + new_w) * (lazy_right - lazy_left) / img_w
            top = y_offset * (lazy_bottom - lazy_top) / img_h
            bottom = (y_offset + new_h) * (lazy_bottom - lazy_top) / img_h
            lazyop['crop_bbox'] = np.array([(lazy_left + left),
                                            (lazy_top + top),
                                            (lazy_left + right),
                                            (lazy_top + bottom)],
                                           dtype=np.float32)

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'input_size={self.input_size}, scales={self.scales}, '
                    f'max_wh_scale_gap={self.max_wh_scale_gap}, '
                    f'random_crop={self.random_crop}, '
                    f'num_fixed_crops={self.num_fixed_crops}, '
                    f'lazy={self.lazy})')
        return repr_str


@PIPELINES.register_module()
class Resize:
    """Resize images to a specific size.

    Required keys are "imgs", "img_shape", "modality", added or modified
    keys are "imgs", "img_shape", "keep_ratio", "scale_factor", "lazy",
    "resize_size". Required keys in "lazy" is None, added or modified key is
    "interpolation".

    Args:
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
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    def __init__(self,
                 scale,
                 keep_ratio=True,
                 interpolation='bilinear',
                 lazy=False):
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
        self.lazy = lazy

    def __call__(self, results):
        """Performs the Resize augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """

        _init_lazy_if_proper(results, self.lazy)

        if 'scale_factor' not in results:
            results['scale_factor'] = np.array([1, 1], dtype=np.float32)
        img_h, img_w = results['img_shape']

        if self.keep_ratio:
            new_w, new_h = mmcv.rescale_size((img_w, img_h), self.scale)
        else:
            new_w, new_h = self.scale

        self.scale_factor = np.array([new_w / img_w, new_h / img_h],
                                     dtype=np.float32)

        results['img_shape'] = (new_h, new_w)
        results['keep_ratio'] = self.keep_ratio
        results['scale_factor'] = results['scale_factor'] * self.scale_factor

        if not self.lazy:
            results['imgs'] = [
                mmcv.imresize(
                    img, (new_w, new_h), interpolation=self.interpolation)
                for img in results['imgs']
            ]
        else:
            lazyop = results['lazy']
            if lazyop['flip']:
                raise NotImplementedError('Put Flip at last for now')
            lazyop['interpolation'] = self.interpolation

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'scale={self.scale}, keep_ratio={self.keep_ratio}, '
                    f'interpolation={self.interpolation}, '
                    f'lazy={self.lazy})')
        return repr_str


@PIPELINES.register_module()
class RandomRescale:
    """Randomly resize images so that the short_edge is resized to a specific
    size in a given range. The scale ratio is unchanged after resizing.

    Required keys are "imgs", "img_shape", "modality", added or modified
    keys are "imgs", "img_shape", "keep_ratio", "scale_factor", "resize_size",
    "short_edge".

    Args:
        scale_range (tuple[int]): The range of short edge length. A closed
            interval.
        interpolation (str): Algorithm used for interpolation:
            "nearest" | "bilinear". Default: "bilinear".
    """

    def __init__(self, scale_range, interpolation='bilinear'):
        self.scale_range = scale_range
        # make sure scale_range is legal, first make sure the type is OK
        assert mmcv.is_tuple_of(scale_range, int)
        assert len(scale_range) == 2
        assert scale_range[0] < scale_range[1]
        assert np.all([x > 0 for x in scale_range])

        self.keep_ratio = True
        self.interpolation = interpolation

    def __call__(self, results):
        """Performs the Resize augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        short_edge = np.random.randint(self.scale_range[0],
                                       self.scale_range[1] + 1)
        resize = Resize((-1, short_edge),
                        keep_ratio=True,
                        interpolation=self.interpolation,
                        lazy=False)
        results = resize(results)

        results['short_edge'] = short_edge
        return results

    def __repr__(self):
        scale_range = self.scale_range
        repr_str = (f'{self.__class__.__name__}('
                    f'scale_range=({scale_range[0]}, {scale_range[1]}), '
                    f'interpolation={self.interpolation})')
        return repr_str


@PIPELINES.register_module()
class Flip:
    """Flip the input images with a probability.

    Reverse the order of elements in the given imgs with a specific direction.
    The shape of the imgs is preserved, but the elements are reordered.
    Required keys are "imgs", "img_shape", "modality", added or modified
    keys are "imgs", "lazy" and "flip_direction". Required keys in "lazy" is
    None, added or modified key are "flip" and "flip_direction".

    Args:
        flip_ratio (float): Probability of implementing flip. Default: 0.5.
        direction (str): Flip imgs horizontally or vertically. Options are
            "horizontal" | "vertical". Default: "horizontal".
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """
    _directions = ['horizontal', 'vertical']

    def __init__(self, flip_ratio=0.5, direction='horizontal', lazy=False):
        if direction not in self._directions:
            raise ValueError(f'Direction {direction} is not supported. '
                             f'Currently support ones are {self._directions}')
        self.flip_ratio = flip_ratio
        self.direction = direction
        self.lazy = lazy

    def __call__(self, results):
        """Performs the Flip augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        _init_lazy_if_proper(results, self.lazy)
        modality = results['modality']
        if modality == 'Flow':
            assert self.direction == 'horizontal'

        flip = np.random.rand() < self.flip_ratio

        results['flip'] = flip
        results['flip_direction'] = self.direction

        if not self.lazy:
            if flip:
                for i, img in enumerate(results['imgs']):
                    mmcv.imflip_(img, self.direction)
                lt = len(results['imgs'])
                for i in range(0, lt, 2):
                    # flow with even indexes are x_flow, which need to be
                    # inverted when doing horizontal flip
                    if modality == 'Flow':
                        results['imgs'][i] = mmcv.iminvert(results['imgs'][i])

            else:
                results['imgs'] = list(results['imgs'])
        else:
            lazyop = results['lazy']
            if lazyop['flip']:
                raise NotImplementedError('Use one Flip please')
            lazyop['flip'] = flip
            lazyop['flip_direction'] = self.direction

        return results

    def __repr__(self):
        repr_str = (
            f'{self.__class__.__name__}('
            f'flip_ratio={self.flip_ratio}, direction={self.direction}, '
            f'lazy={self.lazy})')
        return repr_str


@PIPELINES.register_module()
class Normalize:
    """Normalize images with the given mean and std value.

    Required keys are "imgs", "img_shape", "modality", added or modified
    keys are "imgs" and "img_norm_cfg". If modality is 'Flow', additional
    keys "scale_factor" is required

    Args:
        mean (Sequence[float]): Mean values of different channels.
        std (Sequence[float]): Std values of different channels.
        to_bgr (bool): Whether to convert channels from RGB to BGR.
            Default: False.
        adjust_magnitude (bool): Indicate whether to adjust the flow magnitude
            on 'scale_factor' when modality is 'Flow'. Default: False.
    """

    def __init__(self, mean, std, to_bgr=False, adjust_magnitude=False):
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
        self.adjust_magnitude = adjust_magnitude

    def __call__(self, results):
        modality = results['modality']

        if modality == 'RGB':
            n = len(results['imgs'])
            h, w, c = results['imgs'][0].shape
            imgs = np.empty((n, h, w, c), dtype=np.float32)
            for i, img in enumerate(results['imgs']):
                imgs[i] = img

            for img in imgs:
                mmcv.imnormalize_(img, self.mean, self.std, self.to_bgr)

            results['imgs'] = imgs
            results['img_norm_cfg'] = dict(
                mean=self.mean, std=self.std, to_bgr=self.to_bgr)
            return results
        if modality == 'Flow':
            num_imgs = len(results['imgs'])
            assert num_imgs % 2 == 0
            assert self.mean.shape[0] == 2
            assert self.std.shape[0] == 2
            n = num_imgs // 2
            h, w = results['imgs'][0].shape
            x_flow = np.empty((n, h, w), dtype=np.float32)
            y_flow = np.empty((n, h, w), dtype=np.float32)
            for i in range(n):
                x_flow[i] = results['imgs'][2 * i]
                y_flow[i] = results['imgs'][2 * i + 1]
            x_flow = (x_flow - self.mean[0]) / self.std[0]
            y_flow = (y_flow - self.mean[1]) / self.std[1]
            if self.adjust_magnitude:
                x_flow = x_flow * results['scale_factor'][0]
                y_flow = y_flow * results['scale_factor'][1]
            imgs = np.stack([x_flow, y_flow], axis=-1)
            results['imgs'] = imgs
            args = dict(
                mean=self.mean,
                std=self.std,
                to_bgr=self.to_bgr,
                adjust_magnitude=self.adjust_magnitude)
            results['img_norm_cfg'] = args
            return results
        raise NotImplementedError

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'mean={self.mean}, '
                    f'std={self.std}, '
                    f'to_bgr={self.to_bgr}, '
                    f'adjust_magnitude={self.adjust_magnitude})')
        return repr_str


@PIPELINES.register_module()
class ColorJitter:
    """Randomly distort the brightness, contrast, saturation and hue of images,
    and add PCA based noise into images.

    Note: The input images should be in RGB channel order.

    Code Reference:
    https://gluon-cv.mxnet.io/_modules/gluoncv/data/transforms/experimental/image.html
    https://mxnet.apache.org/api/python/docs/_modules/mxnet/image/image.html#LightingAug

    If specified to apply color space augmentation, it will distort the image
    color space by changing brightness, contrast and saturation. Then, it will
    add some random distort to the images in different color channels.
    Note that the input images should be in original range [0, 255] and in RGB
    channel sequence.

    Required keys are "imgs", added or modified keys are "imgs", "eig_val",
    "eig_vec", "alpha_std" and "color_space_aug".

    Args:
        color_space_aug (bool): Whether to apply color space augmentations. If
            specified, it will change the brightness, contrast, saturation and
            hue of images, then add PCA based noise to images. Otherwise, it
            will directly add PCA based noise to images. Default: False.
        alpha_std (float): Std in the normal Gaussian distribution of alpha.
        eig_val (np.ndarray | None): Eigenvalues of [1 x 3] size for RGB
            channel jitter. If set to None, it will use the default
            eigenvalues. Default: None.
        eig_vec (np.ndarray | None): Eigenvectors of [3 x 3] size for RGB
            channel jitter. If set to None, it will use the default
            eigenvectors. Default: None.
    """

    def __init__(self,
                 color_space_aug=False,
                 alpha_std=0.1,
                 eig_val=None,
                 eig_vec=None):
        if eig_val is None:
            # note that the data range should be [0, 255]
            self.eig_val = np.array([55.46, 4.794, 1.148], dtype=np.float32)
        else:
            self.eig_val = eig_val

        if eig_vec is None:
            self.eig_vec = np.array([[-0.5675, 0.7192, 0.4009],
                                     [-0.5808, -0.0045, -0.8140],
                                     [-0.5836, -0.6948, 0.4203]],
                                    dtype=np.float32)
        else:
            self.eig_vec = eig_vec

        self.alpha_std = alpha_std
        self.color_space_aug = color_space_aug

    @staticmethod
    def brightness(img, delta):
        """Brightness distortion.

        Args:
            img (np.ndarray): An input image.
            delta (float): Delta value to distort brightness.
                It ranges from [-32, 32).

        Returns:
            np.ndarray: A brightness distorted image.
        """
        if np.random.rand() > 0.5:
            img = img + np.float32(delta)
        return img

    @staticmethod
    def contrast(img, alpha):
        """Contrast distortion.

        Args:
            img (np.ndarray): An input image.
            alpha (float): Alpha value to distort contrast.
                It ranges from [0.6, 1.4).

        Returns:
            np.ndarray: A contrast distorted image.
        """
        if np.random.rand() > 0.5:
            img = img * np.float32(alpha)
        return img

    @staticmethod
    def saturation(img, alpha):
        """Saturation distortion.

        Args:
            img (np.ndarray): An input image.
            alpha (float): Alpha value to distort the saturation.
                It ranges from [0.6, 1.4).

        Returns:
            np.ndarray: A saturation distorted image.
        """
        if np.random.rand() > 0.5:
            gray = img * np.array([0.299, 0.587, 0.114], dtype=np.float32)
            gray = np.sum(gray, 2, keepdims=True)
            gray *= (1.0 - alpha)
            img = img * alpha
            img = img + gray
        return img

    @staticmethod
    def hue(img, alpha):
        """Hue distortion.

        Args:
            img (np.ndarray): An input image.
            alpha (float): Alpha value to control the degree of rotation
                for hue. It ranges from [-18, 18).

        Returns:
            np.ndarray: A hue distorted image.
        """
        if np.random.rand() > 0.5:
            u = np.cos(alpha * np.pi)
            w = np.sin(alpha * np.pi)
            bt = np.array([[1.0, 0.0, 0.0], [0.0, u, -w], [0.0, w, u]],
                          dtype=np.float32)
            tyiq = np.array([[0.299, 0.587, 0.114], [0.596, -0.274, -0.321],
                             [0.211, -0.523, 0.311]],
                            dtype=np.float32)
            ityiq = np.array([[1.0, 0.956, 0.621], [1.0, -0.272, -0.647],
                              [1.0, -1.107, 1.705]],
                             dtype=np.float32)
            t = np.dot(np.dot(ityiq, bt), tyiq).T
            t = np.array(t, dtype=np.float32)
            img = np.dot(img, t)
        return img

    def __call__(self, results):
        imgs = results['imgs']
        out = []
        if self.color_space_aug:
            bright_delta = np.random.uniform(-32, 32)
            contrast_alpha = np.random.uniform(0.6, 1.4)
            saturation_alpha = np.random.uniform(0.6, 1.4)
            hue_alpha = np.random.uniform(-18, 18)
            jitter_coin = np.random.rand()
            for img in imgs:
                img = self.brightness(img, delta=bright_delta)
                if jitter_coin > 0.5:
                    img = self.contrast(img, alpha=contrast_alpha)
                    img = self.saturation(img, alpha=saturation_alpha)
                    img = self.hue(img, alpha=hue_alpha)
                else:
                    img = self.saturation(img, alpha=saturation_alpha)
                    img = self.hue(img, alpha=hue_alpha)
                    img = self.contrast(img, alpha=contrast_alpha)
                out.append(img)
        else:
            out = imgs

        # Add PCA based noise
        alpha = np.random.normal(0, self.alpha_std, size=(3, ))
        rgb = np.array(
            np.dot(self.eig_vec * alpha, self.eig_val), dtype=np.float32)
        rgb = rgb[None, None, ...]

        results['imgs'] = [img + rgb for img in out]
        results['eig_val'] = self.eig_val
        results['eig_vec'] = self.eig_vec
        results['alpha_std'] = self.alpha_std
        results['color_space_aug'] = self.color_space_aug

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'color_space_aug={self.color_space_aug}, '
                    f'alpha_std={self.alpha_std}, '
                    f'eig_val={self.eig_val}, '
                    f'eig_vec={self.eig_vec})')
        return repr_str


@PIPELINES.register_module()
class CenterCrop:
    """Crop the center area from images.

    Required keys are "imgs", "img_shape", added or modified keys are "imgs",
    "crop_bbox", "lazy" and "img_shape". Required keys in "lazy" is
    "crop_bbox", added or modified key is "crop_bbox".

    Args:
        crop_size (int | tuple[int]): (w, h) of crop size.
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    def __init__(self, crop_size, lazy=False):
        self.crop_size = _pair(crop_size)
        self.lazy = lazy
        if not mmcv.is_tuple_of(self.crop_size, int):
            raise TypeError(f'Crop_size must be int or tuple of int, '
                            f'but got {type(crop_size)}')

    def __call__(self, results):
        """Performs the CenterCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        _init_lazy_if_proper(results, self.lazy)

        img_h, img_w = results['img_shape']
        crop_w, crop_h = self.crop_size

        left = (img_w - crop_w) // 2
        top = (img_h - crop_h) // 2
        right = left + crop_w
        bottom = top + crop_h
        new_h, new_w = bottom - top, right - left

        results['crop_bbox'] = np.array([left, top, right, bottom])
        results['img_shape'] = (new_h, new_w)

        if not self.lazy:
            results['imgs'] = [
                img[top:bottom, left:right] for img in results['imgs']
            ]
        else:
            lazyop = results['lazy']
            if lazyop['flip']:
                raise NotImplementedError('Put Flip at last for now')

            # record crop_bbox in lazyop dict to ensure only crop once in Fuse
            lazy_left, lazy_top, lazy_right, lazy_bottom = lazyop['crop_bbox']
            left = left * (lazy_right - lazy_left) / img_w
            right = right * (lazy_right - lazy_left) / img_w
            top = top * (lazy_bottom - lazy_top) / img_h
            bottom = bottom * (lazy_bottom - lazy_top) / img_h
            lazyop['crop_bbox'] = np.array([(lazy_left + left),
                                            (lazy_top + top),
                                            (lazy_left + right),
                                            (lazy_top + bottom)],
                                           dtype=np.float32)

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}(crop_size={self.crop_size}, '
                    f'lazy={self.lazy})')
        return repr_str


@PIPELINES.register_module()
class ThreeCrop:
    """Crop images into three crops.

    Crop the images equally into three crops with equal intervals along the
    shorter side.
    Required keys are "imgs", "img_shape", added or modified keys are "imgs",
    "crop_bbox" and "img_shape".

    Args:
        crop_size(int | tuple[int]): (w, h) of crop size.
    """

    def __init__(self, crop_size):
        self.crop_size = _pair(crop_size)
        if not mmcv.is_tuple_of(self.crop_size, int):
            raise TypeError(f'Crop_size must be int or tuple of int, '
                            f'but got {type(crop_size)}')

    def __call__(self, results):
        """Performs the ThreeCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        _init_lazy_if_proper(results, False)

        imgs = results['imgs']
        img_h, img_w = results['imgs'][0].shape[:2]
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

        cropped = []
        crop_bboxes = []
        for x_offset, y_offset in offsets:
            bbox = [x_offset, y_offset, x_offset + crop_w, y_offset + crop_h]
            crop = [
                img[y_offset:y_offset + crop_h, x_offset:x_offset + crop_w]
                for img in imgs
            ]
            cropped.extend(crop)
            crop_bboxes.extend([bbox for _ in range(len(imgs))])

        crop_bboxes = np.array(crop_bboxes)
        results['imgs'] = cropped
        results['crop_bbox'] = crop_bboxes
        results['img_shape'] = results['imgs'][0].shape[:2]

        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}(crop_size={self.crop_size})'
        return repr_str


@PIPELINES.register_module()
class TenCrop:
    """Crop the images into 10 crops (corner + center + flip).

    Crop the four corners and the center part of the image with the same
    given crop_size, and flip it horizontally.
    Required keys are "imgs", "img_shape", added or modified keys are "imgs",
    "crop_bbox" and "img_shape".

    Args:
        crop_size(int | tuple[int]): (w, h) of crop size.
    """

    def __init__(self, crop_size):
        self.crop_size = _pair(crop_size)
        if not mmcv.is_tuple_of(self.crop_size, int):
            raise TypeError(f'Crop_size must be int or tuple of int, '
                            f'but got {type(crop_size)}')

    def __call__(self, results):
        """Performs the TenCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        _init_lazy_if_proper(results, False)

        imgs = results['imgs']

        img_h, img_w = results['imgs'][0].shape[:2]
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
            crop = [
                img[y_offsets:y_offsets + crop_h, x_offset:x_offset + crop_w]
                for img in imgs
            ]
            flip_crop = [np.flip(c, axis=1).copy() for c in crop]
            bbox = [x_offset, y_offsets, x_offset + crop_w, y_offsets + crop_h]
            img_crops.extend(crop)
            img_crops.extend(flip_crop)
            crop_bboxes.extend([bbox for _ in range(len(imgs) * 2)])

        crop_bboxes = np.array(crop_bboxes)
        results['imgs'] = img_crops
        results['crop_bbox'] = crop_bboxes
        results['img_shape'] = results['imgs'][0].shape[:2]

        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}(crop_size={self.crop_size})'
        return repr_str


@PIPELINES.register_module()
class MultiGroupCrop:
    """Randomly crop the images into several groups.

    Crop the random region with the same given crop_size and bounding box
    into several groups.
    Required keys are "imgs", added or modified keys are "imgs", "crop_bbox"
    and "img_shape".

    Args:
        crop_size(int | tuple[int]): (w, h) of crop size.
        groups(int): Number of groups.
    """

    def __init__(self, crop_size, groups):
        self.crop_size = _pair(crop_size)
        self.groups = groups
        if not mmcv.is_tuple_of(self.crop_size, int):
            raise TypeError('Crop size must be int or tuple of int, '
                            f'but got {type(crop_size)}')

        if not isinstance(groups, int):
            raise TypeError(f'Groups must be int, but got {type(groups)}.')

        if groups <= 0:
            raise ValueError('Groups must be positive.')

    def __call__(self, results):
        """Performs the MultiGroupCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        imgs = results['imgs']
        img_h, img_w = imgs[0].shape[:2]
        crop_w, crop_h = self.crop_size

        img_crops = []
        crop_bboxes = []
        for _ in range(self.groups):
            x_offset = random.randint(0, img_w - crop_w)
            y_offset = random.randint(0, img_h - crop_h)

            bbox = [x_offset, y_offset, x_offset + crop_w, y_offset + crop_h]
            crop = [
                img[y_offset:y_offset + crop_h, x_offset:x_offset + crop_w]
                for img in imgs
            ]
            img_crops.extend(crop)
            crop_bboxes.extend([bbox for _ in range(len(imgs))])

        crop_bboxes = np.array(crop_bboxes)
        results['imgs'] = img_crops
        results['crop_bbox'] = crop_bboxes
        results['img_shape'] = results['imgs'][0].shape[:2]

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}'
                    f'(crop_size={self.crop_size}, '
                    f'groups={self.groups})')
        return repr_str


@PIPELINES.register_module()
class AudioAmplify:
    """Amplify the waveform.

    Required keys are "audios", added or modified keys are "audios",
    "amplify_ratio".

    Args:
        ratio (float): The ratio used to amplify the audio waveform.
    """

    def __init__(self, ratio):
        if isinstance(ratio, float):
            self.ratio = ratio
        else:
            raise TypeError('Amplification ratio should be float.')

    def __call__(self, results):
        """Perfrom the audio amplification.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """

        assert 'audios' in results
        results['audios'] *= self.ratio
        results['amplify_ratio'] = self.ratio

        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}(ratio={self.ratio})'
        return repr_str


@PIPELINES.register_module()
class MelSpectrogram:
    """MelSpectrogram. Transfer an audio wave into a melspectogram figure.

    Required keys are "audios", "sample_rate", "num_clips", added or modified
    keys are "audios".

    Args:
        window_size (int): The window size in milisecond. Default: 32.
        step_size (int): The step size in milisecond. Default: 16.
        n_mels (int): Number of mels. Default: 80.
        fixed_length (int): The sample length of melspectrogram maybe not
            exactly as wished due to different fps, fix the length for batch
            collation by truncating or padding. Default: 128.
    """

    def __init__(self,
                 window_size=32,
                 step_size=16,
                 n_mels=80,
                 fixed_length=128):
        if all(
                isinstance(x, int)
                for x in [window_size, step_size, n_mels, fixed_length]):
            self.window_size = window_size
            self.step_size = step_size
            self.n_mels = n_mels
            self.fixed_length = fixed_length
        else:
            raise TypeError('All arguments should be int.')

    def __call__(self, results):
        """Perform MelSpectrogram transformation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        try:
            import librosa
        except ImportError:
            raise ImportError('Install librosa first.')
        signals = results['audios']
        sample_rate = results['sample_rate']
        n_fft = int(round(sample_rate * self.window_size / 1000))
        hop_length = int(round(sample_rate * self.step_size / 1000))
        melspectrograms = list()
        for clip_idx in range(results['num_clips']):
            clip_signal = signals[clip_idx]
            mel = librosa.feature.melspectrogram(
                y=clip_signal,
                sr=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=self.n_mels)
            if mel.shape[0] >= self.fixed_length:
                mel = mel[:self.fixed_length, :]
            else:
                mel = np.pad(
                    mel, ((0, mel.shape[-1] - self.fixed_length), (0, 0)),
                    mode='edge')
            melspectrograms.append(mel)

        results['audios'] = np.array(melspectrograms)
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}'
                    f'(window_size={self.window_size}), '
                    f'step_size={self.step_size}, '
                    f'n_mels={self.n_mels}, '
                    f'fixed_length={self.fixed_length})')
        return repr_str
