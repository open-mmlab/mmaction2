# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from mmaction.evaluation import top_k_accuracy
from mmaction.registry import MODELS
from mmaction.utils import ConfigType, SampleList
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

    def loss_by_feat(self, cls_scores: Union[Tensor, Tuple[Tensor]],
                     data_samples: SampleList) -> dict:
        """Calculate the loss based on the features extracted by the head.

        Args:
            cls_scores (Tensor): Classification prediction results of
                all class, has shape (batch_size, num_classes).
            data_samples (List[:obj:`ActionDataSample`]): The batch
                data samples.

        Returns:
            dict: A dictionary of loss components.
        """
        labels = [x.gt_label for x in data_samples]
        labels = torch.stack(labels).to(cls_scores.device)
        labels = labels.squeeze()

        losses = dict()
        if labels.shape == torch.Size([]):
            labels = labels.unsqueeze(0)
        elif labels.dim() == 1 and cls_scores.size()[0] == 1:
            # Fix a bug when training with soft labels and batch size is 1.
            # When using soft labels, `labels` and `cls_socre` share the same
            # shape.
            labels = labels.unsqueeze(0)

        if cls_scores.size() != labels.size():
            top_k_acc = top_k_accuracy(cls_scores.detach().cpu().numpy(),
                                       labels.detach().cpu().numpy(),
                                       self.topk)
            for k, a in zip(self.topk, top_k_acc):
                losses[f'top{k}_acc'] = torch.tensor(
                    a, device=cls_scores.device)
        if self.label_smooth_eps != 0:
            if cls_scores.size() != labels.size():
                labels = F.one_hot(labels, num_classes=self.num_classes)
            labels = ((1 - self.label_smooth_eps) * labels +
                      self.label_smooth_eps / self.num_classes)

        loss_cls = self.loss_cls(cls_scores, labels)
        # loss_cls may be dictionary or single tensor
        if isinstance(loss_cls, dict):
            losses.update(loss_cls)
        else:
            losses['loss_cls'] = loss_cls
        return losses
