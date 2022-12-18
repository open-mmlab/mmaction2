# Copyright (c) OpenMMLab. All rights reserved.
from torch import Tensor, nn

from mmaction.registry import MODELS
from mmaction.utils import ConfigType
from .base import BaseHead


@MODELS.register_module()
class OmniHead(BaseHead):
    """Classification head for OmniResNet that accepts both image and video
    inputs.

    Args:
        image_classes (int): Number of image classes to be classified.
        video_classes (int): Number of video classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict or ConfigDict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
        image_dropout_ratio (float): Probability of dropout layer for the image
            head. Defaults to 0.2.
        video_dropout_ratio (float): Probability of dropout layer for the video
            head. Defaults to 0.5.
        video_nl_head (bool): if true, use a non-linear head for the video
            head. Defaults to True.
    """

    def __init__(self,
                 image_classes: int,
                 video_classes: int,
                 in_channels: int,
                 loss_cls: ConfigType = dict(type='CrossEntropyLoss'),
                 image_dropout_ratio: float = 0.2,
                 video_dropout_ratio: float = 0.5,
                 video_nl_head: bool = True,
                 **kwargs) -> None:
        super().__init__(image_classes, in_channels, loss_cls, **kwargs)

        self.fc2d = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.BatchNorm1d(in_channels),
            nn.Dropout(image_dropout_ratio),
            nn.Linear(in_channels, image_classes))

        if video_nl_head:
            self.fc3d = nn.Sequential(
                nn.AdaptiveAvgPool3d(1), nn.Flatten(),
                nn.Linear(in_channels, video_classes * 2),
                nn.BatchNorm1d(video_classes * 2), nn.ReLU(inplace=True),
                nn.Dropout(video_dropout_ratio),
                nn.Linear(video_classes * 2, video_classes))
        else:
            self.fc3d = nn.Sequential(
                nn.AdaptiveAvgPool3d(1), nn.Flatten(),
                nn.BatchNorm1d(in_channels), nn.Dropout(video_dropout_ratio),
                nn.Linear(in_channels, video_classes))

    def init_weights(self) -> None:
        pass

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        """Defines the computation performed at every call.

        Args:
            x (Tensor): The input data.

        Returns:
            Tensor: The classification scores for input samples.
        """
        if len(x.shape) == 4:
            cls_score = self.fc2d(x)
        else:
            cls_score = self.fc3d(x)
        return cls_score
