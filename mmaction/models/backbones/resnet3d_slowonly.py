# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence

from mmaction.registry import MODELS
from .resnet3d_slowfast import ResNet3dPathway


@MODELS.register_module()
class ResNet3dSlowOnly(ResNet3dPathway):
    """SlowOnly backbone based on ResNet3dPathway.

    Args:
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
                 conv1_kernel: Sequence[int] = (1, 7, 7),
                 conv1_stride_t: int = 1,
                 pool1_stride_t: int = 1,
                 inflate: Sequence[int] = (0, 0, 1, 1),
                 with_pool2: bool = False,
                 **kwargs) -> None:
        super().__init__(
            conv1_kernel=conv1_kernel,
            conv1_stride_t=conv1_stride_t,
            pool1_stride_t=pool1_stride_t,
            inflate=inflate,
            with_pool2=with_pool2,
            **kwargs)

        assert not self.lateral
