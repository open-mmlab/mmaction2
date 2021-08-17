# Copyright (c) OpenMMLab. All rights reserved.
from collections.abc import Sequence

from mmcv.utils import build_from_cfg

from ..builder import PIPELINES
from .augmentations import PytorchVideoTrans, TorchvisionTrans


@PIPELINES.register_module()
class Compose:
    """Compose a data pipeline with a sequence of transforms.

    Args:
        transforms (list[dict | callable]):
            Either config dicts of transforms or transform objects.
    """

    def __init__(self, transforms):
        assert isinstance(transforms, Sequence)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                if transform['type'].startswith('torchvision.'):
                    trans_type = transform.pop('type')[12:]
                    transform = TorchvisionTrans(trans_type, **transform)
                elif transform['type'].startswith('pytorchvideo.'):
                    trans_type = transform.pop('type')[13:]
                    transform = PytorchVideoTrans(trans_type, **transform)
                else:
                    transform = build_from_cfg(transform, PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError(f'transform must be callable or a dict, '
                                f'but got {type(transform)}')

    def __call__(self, data):
        """Call function to apply transforms sequentially.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
            dict: Transformed data.
        """

        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
