# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModel, BaseModule
from mmengine.runner import CheckpointLoader

from mmaction.registry import MODELS
from mmaction.utils import OptConfigType


def batch_norm(inputs: torch.Tensor,
               module: nn.modules.batchnorm,
               training: Optional[bool] = None) -> torch.Tensor:
    """Applies Batch Normalization for each channel across a batch of data
    using params from the given batch normalization module.

    Args:
        inputs (Tensor): The input data.
        module (nn.modules.batchnorm): a batch normalization module. Will use
            params from this batch normalization module to do the operation.
        training (bool, optional): if true, apply the train mode batch
            normalization. Defaults to None and will use the training mode of
            the module.
    """
    if training is None:
        training = module.training
    return F.batch_norm(
        input=inputs,
        running_mean=None if training else module.running_mean,
        running_var=None if training else module.running_var,
        weight=module.weight,
        bias=module.bias,
        training=training,
        momentum=module.momentum,
        eps=module.eps)


class BottleNeck(BaseModule):
    """Building block for Omni-ResNet.

    Args:
        inplanes (int): Number of channels for the input in first conv layer.
        planes (int): Number of channels for the input in second conv layer.
        temporal_kernel (int): Temporal kernel in the conv layer. Should be
            either 1 or 3. Defaults to 1.
        spatial_stride (int): Spatial stride in the conv layer. Defaults to 1.
        init_cfg (dict or ConfigDict, optional): The Config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 temporal_kernel: int = 3,
                 spatial_stride: int = 1,
                 init_cfg: OptConfigType = None,
                 **kwargs) -> None:
        super(BottleNeck, self).__init__(init_cfg=init_cfg)
        assert temporal_kernel in [1, 3]

        self.conv1 = nn.Conv3d(
            inplanes,
            planes,
            kernel_size=(temporal_kernel, 1, 1),
            padding=(temporal_kernel // 2, 0, 0),
            bias=False)
        self.conv2 = nn.Conv3d(
            planes,
            planes,
            stride=(1, spatial_stride, spatial_stride),
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
            bias=False)

        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm3d(planes, momentum=0.01)
        self.bn2 = nn.BatchNorm3d(planes, momentum=0.01)
        self.bn3 = nn.BatchNorm3d(planes * 4, momentum=0.01)

        if inplanes != planes * 4 or spatial_stride != 1:
            downsample = [
                nn.Conv3d(
                    inplanes,
                    planes * 4,
                    kernel_size=1,
                    stride=(1, spatial_stride, spatial_stride),
                    bias=False),
                nn.BatchNorm3d(planes * 4, momentum=0.01)
            ]
            self.downsample = nn.Sequential(*downsample)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call.

        Accept both 3D (BCTHW for videos) and 2D (BCHW for images) tensors.
        """
        if x.ndim == 4:
            return self.forward_2d(x)

        # Forward call for 3D tensors.
        out = self.conv1(x)
        out = self.bn1(out).relu_()

        out = self.conv2(out)
        out = self.bn2(out).relu_()

        out = self.conv3(out)
        out = self.bn3(out)

        if hasattr(self, 'downsample'):
            x = self.downsample(x)

        return out.add_(x).relu_()

    def forward_2d(self, x: torch.Tensor) -> torch.Tensor:
        """Forward call for 2D tensors."""
        out = F.conv2d(x, self.conv1.weight.sum(2))
        out = batch_norm(out, self.bn1).relu_()

        out = F.conv2d(
            out,
            self.conv2.weight.squeeze(2),
            stride=self.conv2.stride[-1],
            padding=1)
        out = batch_norm(out, self.bn2).relu_()

        out = F.conv2d(out, self.conv3.weight.squeeze(2))
        out = batch_norm(out, self.bn3)

        if hasattr(self, 'downsample'):
            x = F.conv2d(
                x,
                self.downsample[0].weight.squeeze(2),
                stride=self.downsample[0].stride[-1])
            x = batch_norm(x, self.downsample[1])

        return out.add_(x).relu_()


@MODELS.register_module()
class OmniResNet(BaseModel):
    """Omni-ResNet that accepts both image and video inputs.

    Args:
        layers (List[int]): number of layers in each residual stages. Defaults
            to [3, 4, 6, 3].
        pretrain_2d (str, optional): path to the 2D pretraining checkpoints.
            Defaults to None.
        init_cfg (dict or ConfigDict, optional): The Config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 layers: List[int] = [3, 4, 6, 3],
                 pretrain_2d: Optional[str] = None,
                 init_cfg: OptConfigType = None) -> None:
        super(OmniResNet, self).__init__(init_cfg=init_cfg)

        self.inplanes = 64
        self.conv1 = nn.Conv3d(
            3,
            self.inplanes,
            kernel_size=(1, 7, 7),
            stride=(1, 2, 2),
            padding=(0, 3, 3),
            bias=False)
        self.bn1 = nn.BatchNorm3d(self.inplanes, momentum=0.01)

        self.pool3d = nn.MaxPool3d((1, 3, 3), (1, 2, 2), (0, 1, 1))
        self.pool2d = nn.MaxPool2d(3, 2, 1)

        self.temporal_kernel = 1
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.temporal_kernel = 3
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)

        if pretrain_2d is not None:
            self.init_from_2d(pretrain_2d)

    def _make_layer(self,
                    planes: int,
                    num_blocks: int,
                    stride: int = 1) -> nn.Module:
        layers = [
            BottleNeck(
                self.inplanes,
                planes,
                spatial_stride=stride,
                temporal_kernel=self.temporal_kernel)
        ]
        self.inplanes = planes * 4
        for _ in range(1, num_blocks):
            layers.append(
                BottleNeck(
                    self.inplanes,
                    planes,
                    temporal_kernel=self.temporal_kernel))
        return nn.Sequential(*layers)

    def init_from_2d(self, pretrain: str) -> None:
        param2d = CheckpointLoader.load_checkpoint(
            pretrain, map_location='cpu')
        param3d = self.state_dict()
        for key in param3d:
            if key in param2d:
                weight = param2d[key]
                if weight.ndim == 4:
                    t = param3d[key].shape[2]
                    weight = weight.unsqueeze(2)
                    weight = weight.expand(-1, -1, t, -1, -1)
                    weight = weight / t
                param3d[key] = weight
        self.load_state_dict(param3d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call.

        Accept both 3D (BCTHW for videos) and 2D (BCHW for images) tensors.
        """
        if x.ndim == 4:
            return self.forward_2d(x)

        # Forward call for 3D tensors.
        x = self.conv1(x)
        x = self.bn1(x).relu_()
        x = self.pool3d(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def forward_2d(self, x: torch.Tensor) -> torch.Tensor:
        """Forward call for 2D tensors."""
        x = F.conv2d(
            x,
            self.conv1.weight.squeeze(2),
            stride=self.conv1.stride[-1],
            padding=self.conv1.padding[-1])
        x = batch_norm(x, self.bn1).relu_()
        x = self.pool2d(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
