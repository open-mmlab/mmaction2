# adapted from basicTAD
# https://github.com/open-mmlab/mmcv or
# https://github.com/open-mmlab/mmdetection

from typing import Sequence

import mmcv
import numpy as np
from mmcv.transforms import to_tensor
from mmcv.transforms.base import BaseTransform
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData
from numpy import random

from mmaction.registry import TRANSFORMS
from ..models.task_modules.segments_ops import segment_overlaps


@TRANSFORMS.register_module()
class Time2Frame(BaseTransform):
    """Switch time point to frame index."""

    def transform(self, results):
        results['segments'] = results['segments'] * results['fps']

        return results


@TRANSFORMS.register_module()
class TemporalRandomCrop(BaseTransform):
    """Temporally crop.

    Args:
        clip_len (int, optional): The cropped frame num. Default: 768.
        iof_th(float, optional): The minimal iof threshold to crop. Default: 0
    """

    def __init__(self, clip_len=96, frame_interval=10, iof_th=0.75):
        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.iof_th = iof_th

    def get_valid_mask(self, segments, patch, iof_th):
        gt_iofs = segment_overlaps(segments, patch, mode='iof')[:, 0]
        patch_iofs = segment_overlaps(patch, segments, mode='iof')[0, :]
        iofs = np.maximum(gt_iofs, patch_iofs)
        mask = iofs > iof_th

        return mask

    def transform(self, results):
        """Call function to random temporally crop video frame.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Temporally cropped results, 'frame_inds' is updated in
                result dict.
        """
        total_frames = results['total_frames']
        ori_clip_len = (self.clip_len - 1) * self.frame_interval + 1
        ori_clip_len = min(ori_clip_len, total_frames)
        while True:
            clip = np.arange(self.clip_len) * self.frame_interval
            offset = np.random.randint(0, total_frames - ori_clip_len + 1)
            clip = clip + offset
            clip = clip[clip < total_frames]
            start, end = clip[0], clip[-1]

            segments = results['segments']
            mask = self.get_valid_mask(segments, np.array([[start, end]], dtype=np.float32), self.iof_th)

            # If the cropped clip does NOT have IoF greater than the threshold with any (acknowledged) actions, then re-crop.
            if not np.logical_and(mask, np.logical_not(results['ignore_flags'])).any():
                continue

            segments = segments[mask]
            segments = segments.clip(min=start, max=end)  # TODO: Is this necessary?
            segments -= start  # transform the index of segments to be relative to the cropped segment
            segments = segments / self.frame_interval  # to be relative to the input clip
            assert segments.max() < len(clip)
            assert segments.min() >= 0

            results['segments'] = segments
            results['labels'] = results['labels'][mask]
            results['ignore_flags'] = results['ignore_flags'][mask]
            results['frame_inds'] = clip
            assert max(results['frame_inds']) < total_frames, f"offset: {offset}\n" \
                                                              f"start, end: [{start}, {end}]," \
                                                              f"total frames: {total_frames}"
            results['num_clips'] = 1
            results['clip_len'] = self.clip_len
            results['tsize'] = len(clip)

            if 'img_idx_mapping' in results:
                results['frame_inds'] = results['img_idx_mapping'][clip]
                assert results['frame_inds'].max() < results['total_frames']
                assert results['frame_inds'].min() >= 0
            return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(clip_len={self.clip_len},'
        repr_str += f'(frame_interval={self.frame_interval},'
        repr_str += f'iof_th={self.iof_th})'

        return repr_str


@TRANSFORMS.register_module()
class SpatialRandomCrop(BaseTransform):
    """Spatially random crop images.
    Args:
        crop_size (tuple): Expected size after cropping, (h, w).
    Notes:
        - If the image is smaller than the crop size, return the original image
    """

    def __init__(self, crop_size):
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size

    def transform(self, results):
        """Call function to randomly crop images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Randomly cropped results, 'imgs_shape' key in result dict
                is updated according to crop size.
        """
        img_h, img_w = results['img_shape']
        margin_h = max(img_h - self.crop_size[0], 0)
        margin_w = max(img_w - self.crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

        # crop images
        imgs = [img[crop_y1:crop_y2, crop_x1:crop_x2] for img in results['imgs']]
        results['imgs'] = imgs

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'


@TRANSFORMS.register_module()
class PhotoMetricDistortion(BaseTransform):
    """Apply photometric distortion to images sequentially, every
    transformation is applied with a probability of 0.5. The position of random
    contrast is in second or second to last.

    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18,
                 p=0.5):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta
        self.p = p

    def transform(self, results):
        """Call function to perform photometric distortion on images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images distorted.
        """

        imgs = np.array(results['imgs']).astype(np.float32)

        def _filter(img):
            img[img < 0] = 0
            img[img > 255] = 255
            return img

        if random.uniform(0, 1) <= self.p:

            # random brightness
            if random.randint(2):
                delta = random.uniform(-self.brightness_delta,
                                       self.brightness_delta)
                imgs += delta
                imgs = _filter(imgs)

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = random.randint(2)
            if mode == 1:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower,
                                           self.contrast_upper)
                    imgs *= alpha
                    imgs = _filter(imgs)

            # convert color from BGR to HSV
            imgs = np.array([mmcv.image.bgr2hsv(img) for img in imgs])

            # random saturation
            if random.randint(2):
                imgs[..., 1] *= random.uniform(self.saturation_lower,
                                               self.saturation_upper)

            # random hue
            # if random.randint(2):
            if True:
                imgs[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
                imgs[..., 0][imgs[..., 0] > 360] -= 360
                imgs[..., 0][imgs[..., 0] < 0] += 360

            # convert color from HSV to BGR
            imgs = np.array([mmcv.image.hsv2bgr(img) for img in imgs])
            imgs = _filter(imgs)

            # random contrast
            if mode == 0:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower,
                                           self.contrast_upper)
                    imgs *= alpha
                    imgs = _filter(imgs)

            # randomly swap channels
            if random.randint(2):
                imgs = imgs[..., random.permutation(3)]

            results['imgs'] = list(imgs)  # change back to mmaction-style (list of) imgs
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(\nbrightness_delta={self.brightness_delta},\n'
        repr_str += 'contrast_range='
        repr_str += f'{(self.contrast_lower, self.contrast_upper)},\n'
        repr_str += 'saturation_range='
        repr_str += f'{(self.saturation_lower, self.saturation_upper)},\n'
        repr_str += f'hue_delta={self.hue_delta})'
        return repr_str


@TRANSFORMS.register_module()
class Rotate(BaseTransform):
    """Spatially rotate images.

    Args:
        limit (int, list or tuple): Angle range, (min_angle, max_angle).
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos".
            Default: bilinear
        border_mode (str): Border mode, accepted values are "constant",
            "isolated", "reflect", "reflect_101", "replicate", "transparent",
            "wrap". Default: constant
        border_value (int): Border value. Default: 0
    """

    def __init__(self,
                 limit,
                 interpolation='bilinear',
                 border_mode='constant',
                 border_value=0,
                 p=0.5):
        if isinstance(limit, int):
            limit = (-limit, limit)
        self.limit = limit
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.border_value = border_value
        self.p = p

    def transform(self, results):
        """Call function to random rotate images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Spatially rotated results.
        """

        if random.uniform(0, 1) <= self.p:
            angle = random.uniform(*self.limit)
            imgs = [
                mmcv.image.imrotate(
                    img,
                    angle=angle,
                    interpolation=self.interpolation,
                    border_mode=self.border_mode,
                    border_value=self.border_value) for img in results['imgs']]

            results['imgs'] = [np.ascontiguousarray(img) for img in imgs]

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(limit={self.limit},'
        repr_str += f'interpolation={self.interpolation},'
        repr_str += f'border_mode={self.border_mode},'
        repr_str += f'border_value={self.border_value},'
        repr_str += f'p={self.p})'

        return repr_str


@TRANSFORMS.register_module()
class Pad(BaseTransform):
    """Pad images.

    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",

    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    @staticmethod
    def impad(img, shape, pad_val=0):
        """Pad an image or images to a certain shape.
        Args:
            img (ndarray): Image to be padded.
            shape (tuple[int]): Expected padding shape (h, w).
            pad_val (Number | Sequence[Number]): Values to be filled in padding
                areas. Default: 0.
        Returns:
            ndarray: The padded image.
        """
        if not isinstance(pad_val, (int, float)):
            assert len(pad_val) == img.shape[-1]
        if len(shape) < len(img.shape):
            shape = shape + (img.shape[-1],)
        assert len(shape) == len(img.shape)
        for s, img_s in zip(shape, img.shape):
            assert s >= img_s, f"pad shape {s} should be greater than image shape {img_s}"
        pad = np.empty(shape, dtype=img.dtype)
        pad[...] = pad_val
        pad[:img.shape[0], :img.shape[1], :img.shape[2], ...] = img
        return pad

    @staticmethod
    def impad_to_multiple(img, divisor, pad_val=0):
        """Pad an image to ensure each edge to be multiple to some number.
        Args:
            img (ndarray): Image to be padded.
            divisor (int): Padded image edges will be multiple to divisor.
            pad_val (Number | Sequence[Number]): Same as :func:`impad`.
        Returns:
            ndarray: The padded image.
        """
        pad_shape = tuple(
            int(np.ceil(shape / divisor)) * divisor for shape in img.shape[:-1])
        return Pad.impad(img, pad_shape, pad_val)

    def transform(self, results):
        """Call function to pad images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """
        if self.size is not None:
            padded_imgs = self.impad(
                np.array(results['imgs']), shape=self.size, pad_val=self.pad_val)
        elif self.size_divisor is not None:
            padded_imgs = self.impad_to_multiple(
                np.array(results['imgs']), self.size_divisor, pad_val=self.pad_val)
        else:
            raise AssertionError("Either 'size' or 'size_divisor' need to be set, but both None")
        results['imgs'] = list(padded_imgs)  # change back to mmaction-style (list of) imgs
        results['pad_tsize'] = padded_imgs.shape[0]

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str


@TRANSFORMS.register_module()
class PackTadInputs(BaseTransform):
    """Pack the inputs data for the detection / semantic segmentation /
    panoptic segmentation.

    The ``img_meta`` item is always populated.  The contents of the
    ``img_meta`` dictionary depends on ``meta_keys``. By default this includes:

        - ``img_id``: id of the image

        - ``img_path``: path to the image file

        - ``ori_shape``: original shape of the image as a tuple (h, w)

        - ``img_shape``: shape of the image input to the network as a tuple \
            (h, w).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.

        - ``scale_factor``: a float indicating the preprocessing scale

        - ``flip``: a boolean indicating if image flip transform was used

        - ``flip_direction``: the flipping direction

    Args:
        meta_keys (Sequence[str], optional): Meta keys to be converted to
            ``mmcv.DataContainer`` and collected in ``data[img_metas]``.
            Default: ``('img_id', 'img_path', 'ori_shape', 'img_shape',
            'scale_factor', 'flip', 'flip_direction')``
    """

    def __init__(self,
                 meta_keys=('img_id', 'img_shape', 'scale_factor')):
        self.meta_keys = meta_keys

    @staticmethod
    def mmdet_mapping(results: dict) -> dict:
        # Modify the meta keys/values to be consistent with mmdet
        results['img'] = results['imgs']
        results['img_shape'] = (1, results.pop('tsize'))
        results['pad_shape'] = (1, results.pop('pad_tsize'))
        if 'tscale_factor' in results:
            results['scale_factor'] = (results.pop('tscale_factor'), 1)  # (w, h)
        results['img_id'] = results.pop('video_name')

        gt_bboxes = np.insert(results['segments'], 2, 0.9, axis=-1)
        gt_bboxes = np.insert(gt_bboxes, 1, 0.1, axis=-1)
        results['bboxes'] = gt_bboxes
        results['labels'] = results.pop('labels')

        return results

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict:

            - 'inputs' (obj:`torch.Tensor`): The forward data of models.
            - 'data_sample' (obj:`DetDataSample`): The annotation info of the
                sample.
        """
        results = self.mmdet_mapping(results)
        packed_results = dict()

        img = results['img']
        if not img.flags.c_contiguous:
            img = to_tensor(np.ascontiguousarray(img))
        else:
            img = to_tensor(img).contiguous()

        packed_results['inputs'] = img

        data_sample = DetDataSample(gt_instances=InstanceData(bboxes=to_tensor(results['bboxes']),
                                                              labels=to_tensor(results['labels'])))
        img_meta = {}
        for key in self.meta_keys:
            assert key in results, f'`{key}` is not found in `results`, ' \
                                   f'the valid keys are {list(results)}.'
            img_meta[key] = results[key]

        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample

        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str


@TRANSFORMS.register_module()
class SpatialCenterCrop(BaseTransform):
    """Spatially center crop images.

    Args:
        crop_size (tuple): Expected size after cropping, (h, w).

    Notes:
        - If the image is smaller than the crop size, return the original image
    """

    def __init__(self, crop_size):
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size

    def transform(self, results):
        """Call function to center crop images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'imgs_shape' key in result dict
                is updated according to crop size.
        """

        imgs = np.array(results['imgs'])
        margin_h = max(imgs.shape[1] - self.crop_size[0], 0)
        margin_w = max(imgs.shape[2] - self.crop_size[1], 0)
        offset_h = int(margin_h / 2)
        offset_w = int(margin_w / 2)
        crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

        # crop images
        imgs = imgs[:, crop_y1:crop_y2, crop_x1:crop_x2, ...]
        results['imgs'] = list(imgs)

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'
