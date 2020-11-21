from collections.abc import Sequence

import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer as DC

from ..registry import PIPELINES


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return data
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    if isinstance(data, Sequence) and not mmcv.is_str(data):
        return torch.tensor(data)
    if isinstance(data, int):
        return torch.LongTensor([data])
    if isinstance(data, float):
        return torch.FloatTensor([data])
    raise TypeError(f'type {type(data)} cannot be converted to tensor.')


@PIPELINES.register_module()
class ToTensor:
    """Convert some values in results dict to `torch.Tensor` type in data
    loader pipeline.

    Args:
        keys (Sequence[str]): Required keys to be converted.
    """

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        """Performs the ToTensor formating.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        for key in self.keys:
            results[key] = to_tensor(results[key])
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(keys={self.keys})'


@PIPELINES.register_module()
class ToDataContainer:
    """Convert the data to DataContainer.

    Args:
        fields (Sequence[dict]): Required fields to be converted
            with keys and attributes. E.g.
            fields=(dict(key='gt_bbox', stack=False),).
    """

    def __init__(self, fields):
        self.fields = fields

    def __call__(self, results):
        """Performs the ToDataContainer formating.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        for field in self.fields:
            _field = field.copy()
            key = _field.pop('key')
            results[key] = DC(results[key], **_field)
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(fields={self.fields})'


@PIPELINES.register_module()
class ImageToTensor:
    """Convert image type to `torch.Tensor` type.

    Args:
        keys (Sequence[str]): Required keys to be converted.
    """

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        """Performs the ImageToTensor formating.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        for key in self.keys:
            results[key] = to_tensor(results[key].transpose(2, 0, 1))
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(keys={self.keys})'


@PIPELINES.register_module()
class Transpose:
    """Transpose image channels to a given order.

    Args:
        keys (Sequence[str]): Required keys to be converted.
        order (Sequence[int]): Image channel order.
    """

    def __init__(self, keys, order):
        self.keys = keys
        self.order = order

    def __call__(self, results):
        """Performs the Transpose formatting.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        for key in self.keys:
            results[key] = results[key].transpose(self.order)
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'keys={self.keys}, order={self.order})')


@PIPELINES.register_module()
class Collect:
    """Collect data from the loader relevant to the specific task.

    This keeps the items in ``keys`` as it is, and collect items in
    ``meta_keys`` into a meta item called ``meta_name``.This is usually
    the last stage of the data loader pipeline.
    For example, when keys='imgs', meta_keys=('filename', 'label',
    'original_shape'), meta_name='img_meta', the results will be a dict with
    keys 'imgs' and 'img_meta', where 'img_meta' is a DataContainer of another
    dict with keys 'filename', 'label', 'original_shape'.

    Args:
        keys (Sequence[str]): Required keys to be collected.
        meta_name (str): The name of the key that contains meta infomation.
            This key is always populated. Default: "img_meta".
        meta_keys (Sequence[str]): Keys that are collected under meta_name.
            The contents of the ``meta_name`` dictionary depends on
            ``meta_keys``.
            By default this includes:

            - "filename": path to the image file
            - "label": label of the image file
            - "original_shape": original shape of the image as a tuple
                (h, w, c)
            - "img_shape": shape of the image input to the network as a tuple
                (h, w, c).  Note that images may be zero padded on the
                bottom/right, if the batch tensor is larger than this shape.
            - "pad_shape": image shape after padding
            - "flip_direction": a str in ("horiziontal", "vertival") to
                indicate if the image is fliped horizontally or vertically.
            - "img_norm_cfg": a dict of normalization information:
                - mean - per channel mean subtraction
                - std - per channel std divisor
                - to_rgb - bool indicating if bgr was converted to rgb
    """

    def __init__(self,
                 keys,
                 meta_keys=('filename', 'label', 'original_shape', 'img_shape',
                            'pad_shape', 'flip_direction', 'img_norm_cfg'),
                 meta_name='img_meta'):
        self.keys = keys
        self.meta_keys = meta_keys
        self.meta_name = meta_name

    def __call__(self, results):
        """Performs the Collect formating.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        data = {}
        for key in self.keys:
            data[key] = results[key]

        if len(self.meta_keys) != 0:
            meta = {}
            for key in self.meta_keys:
                meta[key] = results[key]
            data[self.meta_name] = DC(meta, cpu_only=True)

        return data

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'keys={self.keys}, meta_keys={self.meta_keys})')


@PIPELINES.register_module()
class FormatShape:
    """Format final imgs shape to the given input_format.

    Required keys are "imgs", "num_clips" and "clip_len", added or modified
    keys are "imgs" and "input_shape".

    Args:
        input_format (str): Define the final imgs format.
    """

    def __init__(self, input_format):
        self.input_format = input_format
        if self.input_format not in ['NCTHW', 'NCHW', 'NCHW_Flow', 'NPTCHW']:
            raise ValueError(
                f'The input format {self.input_format} is invalid.')

    def __call__(self, results):
        """Performs the FormatShape formating.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        imgs = results['imgs']
        # [M x H x W x C]
        # M = 1 * N_crops * N_clips * L
        if self.input_format == 'NCTHW':
            num_clips = results['num_clips']
            clip_len = results['clip_len']

            imgs = imgs.reshape((-1, num_clips, clip_len) + imgs.shape[1:])
            # N_crops x N_clips x L x H x W x C
            imgs = np.transpose(imgs, (0, 1, 5, 2, 3, 4))
            # N_crops x N_clips x C x L x H x W
            imgs = imgs.reshape((-1, ) + imgs.shape[2:])
            # M' x C x L x H x W
            # M' = N_crops x N_clips
        elif self.input_format == 'NCHW':
            imgs = np.transpose(imgs, (0, 3, 1, 2))
            # M x C x H x W
        elif self.input_format == 'NCHW_Flow':
            num_clips = results['num_clips']
            clip_len = results['clip_len']
            imgs = imgs.reshape((-1, num_clips, clip_len) + imgs.shape[1:])
            # N_crops x N_clips x L x H x W x C
            imgs = np.transpose(imgs, (0, 1, 2, 5, 3, 4))
            # N_crops x N_clips x L x C x H x W
            imgs = imgs.reshape((-1, imgs.shape[2] * imgs.shape[3]) +
                                imgs.shape[4:])
            # M' x C' x H x W
            # M' = N_crops x N_clips
            # C' = L x C
        elif self.input_format == 'NPTCHW':
            num_proposals = results['num_proposals']
            num_clips = results['num_clips']
            clip_len = results['clip_len']
            imgs = imgs.reshape((num_proposals, num_clips * clip_len) +
                                imgs.shape[1:])
            # P x M x H x W x C
            # M = N_clips x L
            imgs = np.transpose(imgs, (0, 1, 4, 2, 3))
            # P x M x C x H x W

        results['imgs'] = imgs
        results['input_shape'] = imgs.shape
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(input_format='{self.input_format}')"
        return repr_str


@PIPELINES.register_module()
class FormatAudioShape:
    """Format final audio shape to the given input_format.

    Required keys are "imgs", "num_clips" and "clip_len", added or modified
    keys are "imgs" and "input_shape".

    Args:
        input_format (str): Define the final imgs format.
    """

    def __init__(self, input_format):
        self.input_format = input_format
        if self.input_format not in ['NCTF']:
            raise ValueError(
                f'The input format {self.input_format} is invalid.')

    def __call__(self, results):
        """Performs the FormatShape formatting.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        audios = results['audios']
        # clip x sample x freq -> clip x channel x sample x freq
        clip, sample, freq = audios.shape
        audios = audios.reshape(clip, 1, sample, freq)
        results['audios'] = audios
        results['input_shape'] = audios.shape
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(input_format='{self.input_format}')"
        return repr_str
