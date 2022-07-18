# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence

from mmaction.registry import MODELS
from .resnet3d_slowfast import ResNet3dPathway

try:
    from mmdet.registry import MODELS as MMDET_MODELS
    mmdet_imported = True
except (ImportError, ModuleNotFoundError):
    mmdet_imported = False


@MODELS.register_module()
class ResNet3dSlowOnly(ResNet3dPathway):
    """SlowOnly backbone based on ResNet3dPathway.

    Args:
        lateral (bool): Determines whether to enable the lateral connection
            from another pathway. Defaults to False.
        conv1_kernel (Sequence[int]): Kernel size of the first conv layer.
            Defaults to ``(1, 7, 7)``.
        conv1_stride_t (int): Temporal stride of the first conv layer.
            Defaults to 1.
        pool1_stride_t (int): Temporal stride of the first pooling layer.
            Defaults to 1.
        inflate (Sequence[int]): Inflate dims of each block.
            Defaults to ``(0, 0, 1, 1)``.
        with_pool2 (bool): Whether to use pool2. Defaults to False.
    """

    def __init__(self,
                 *args,
                 lateral: bool = False,
                 conv1_kernel: Sequence[int] = (1, 7, 7),
                 conv1_stride_t: int = 1,
                 pool1_stride_t: int = 1,
                 inflate: Sequence[int] = (0, 0, 1, 1),
                 with_pool2: bool = False,
                 **kwargs) -> None:
        super().__init__(
            *args,
            lateral=lateral,
            conv1_kernel=conv1_kernel,
            conv1_stride_t=conv1_stride_t,
            pool1_stride_t=pool1_stride_t,
            inflate=inflate,
            with_pool2=with_pool2,
            **kwargs)

        assert not self.lateral


if mmdet_imported:
    MMDET_MODELS.register_module()(ResNet3dSlowOnly)
