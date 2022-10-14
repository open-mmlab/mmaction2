# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple, Union

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmaction.models.backbones.resnet import ResNet
from mmaction.registry import MODELS


@MODELS.register_module()
class C2D(ResNet):
    """C2D backbone.

    Compared to ResNet-50, a temporal-pool is added after the first
    bottleneck. Detailed structure is kept same as "video-nonlocal-net" repo.
    Please refer to https://github.com/facebookresearch/video-nonlocal-net/blob
    /main/scripts/run_c2d_baseline_400k.sh.
    Please note that there are some improvements compared to "Non-local Neural
    Networks" paper (https://arxiv.org/abs/1711.07971).
    Differences are noted at https://github.com/facebookresearch/video-nonlocal
    -net#modifications-for-improving-speed.
    """

    def _make_stem_layer(self) -> None:
        """Construct the stem layers consists of a conv+norm+act module and a
        pooling layer."""
        self.conv1 = ConvModule(
            self.in_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.maxpool3d_1 = nn.MaxPool3d(
            kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 0, 0))
        self.maxpool3d_2 = nn.MaxPool3d(
            kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))

    def forward(self, x: torch.Tensor) \
            -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            Union[torch.Tensor or Tuple[torch.Tensor]]: The feature of the
                input samples extracted by the backbone.
        """

        batches = x.shape[0]

        def _convert_to_2d(x: torch.Tensor) -> torch.Tensor:
            """(N, C, T, H, W) -> (N x T, C, H, W)"""
            x = x.permute((0, 2, 1, 3, 4))
            x = x.reshape(-1, x.shape[2], x.shape[3], x.shape[4])
            return x

        def _convert_to_3d(x: torch.Tensor) -> torch.Tensor:
            """(N x T, C, H, W) -> (N, C, T, H, W)"""
            x = x.reshape(batches, -1, x.shape[1], x.shape[2], x.shape[3])
            x = x.permute((0, 2, 1, 3, 4))
            return x

        x = _convert_to_2d(x)
        x = self.conv1(x)
        x = _convert_to_3d(x)
        x = self.maxpool3d_1(x)
        x = _convert_to_2d(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i == 0:
                x = _convert_to_3d(x)
                x = self.maxpool3d_2(x)
                x = _convert_to_2d(x)
            if i in self.out_indices:
                x = _convert_to_3d(x)
                outs.append(x)
        if len(outs) == 1:
            return outs[0]

        return tuple(outs)
