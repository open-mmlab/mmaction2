# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple, Union

import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmengine.model.weight_init import constant_init, kaiming_init
from torch.nn.modules.utils import _triple

from mmaction.registry import MODELS
from mmaction.utils import ConfigType


@MODELS.register_module()
class Conv2plus1d(nn.Module):
    """(2+1)d Conv module for R(2+1)d backbone.

    https://arxiv.org/pdf/1711.11248.pdf.

    Args:
        in_channels (int): Same as ``nn.Conv3d``.
        out_channels (int): Same as ``nn.Conv3d``.
        kernel_size (Union[int, Tuple[int]]): Same as ``nn.Conv3d``.
        stride (Union[int, Tuple[int]]): Same as ``nn.Conv3d``. Defaults to 1.
        padding (Union[int, Tuple[int]]): Same as ``nn.Conv3d``. Defaults to 0.
        dilation (Union[int, Tuple[int]]): Same as ``nn.Conv3d``.
            Defaults to 1.
        groups (int): Same as ``nn.Conv3d``. Defaults to 1.
        bias (Union[bool, str]): If specified as `auto`, it will be decided by
            the norm_cfg. Bias will be set as True if norm_cfg is None,
            otherwise False.
        norm_cfg (Union[dict, ConfigDict]): Config for norm layers.
            Defaults to ``dict(type='BN3d')``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int]],
        stride: Union[int, Tuple[int]] = 1,
        padding: Union[int, Tuple[int]] = 0,
        dilation: Union[int, Tuple[int]] = 1,
        groups: int = 1,
        bias: Union[bool, str] = True,
        norm_cfg: ConfigType = dict(type='BN3d')
    ) -> None:
        super().__init__()

        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        assert len(kernel_size) == len(stride) == len(padding) == 3

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.norm_cfg = norm_cfg
        self.output_padding = (0, 0, 0)
        self.transposed = False

        # The middle-plane is calculated according to:
        # M_i = \floor{\frac{t * d^2 N_i-1 * N_i}
        #   {d^2 * N_i-1 + t * N_i}}
        # where d, t are spatial and temporal kernel, and
        # N_i, N_i-1 are planes
        # and inplanes. https://arxiv.org/pdf/1711.11248.pdf
        mid_channels = 3 * (
            in_channels * out_channels * kernel_size[1] * kernel_size[2])
        mid_channels /= (
            in_channels * kernel_size[1] * kernel_size[2] + 3 * out_channels)
        mid_channels = int(mid_channels)

        self.conv_s = nn.Conv3d(
            in_channels,
            mid_channels,
            kernel_size=(1, kernel_size[1], kernel_size[2]),
            stride=(1, stride[1], stride[2]),
            padding=(0, padding[1], padding[2]),
            bias=bias)
        _, self.bn_s = build_norm_layer(self.norm_cfg, mid_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv_t = nn.Conv3d(
            mid_channels,
            out_channels,
            kernel_size=(kernel_size[0], 1, 1),
            stride=(stride[0], 1, 1),
            padding=(padding[0], 0, 0),
            bias=bias)

        self.init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The output of the module.
        """
        x = self.conv_s(x)
        x = self.bn_s(x)
        x = self.relu(x)
        x = self.conv_t(x)
        return x

    def init_weights(self) -> None:
        """Initiate the parameters from scratch."""
        kaiming_init(self.conv_s)
        kaiming_init(self.conv_t)
        constant_init(self.bn_s, 1, bias=0)
