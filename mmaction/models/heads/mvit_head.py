# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

from mmengine.model.weight_init import constant_init, trunc_normal_init
from torch import Tensor, nn

from mmaction.registry import MODELS
from mmaction.utils import ConfigType
from .base import BaseHead


@MODELS.register_module()
class MViTHead(BaseHead):
    """Classification head for Multi-scale ViT.

    A PyTorch implement of : `MViTv2: Improved Multiscale Vision Transformers
    for Classification and Detection <https://arxiv.org/abs/2112.01526>`_

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict or ConfigDict): Config for building loss.
            Defaults to `dict(type='CrossEntropyLoss')`.
        dropout_ratio (float): Probability of dropout layer. Defaults to 0.5.
        init_std (float): Std value for Initiation. Defaults to 0.02.
        init_scale (float): Scale factor for Initiation parameters.
            Defaults to 1.
        with_cls_token (bool): Whether the backbone output feature with
            cls_token. Defaults to True.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 loss_cls: ConfigType = dict(type='CrossEntropyLoss'),
                 dropout_ratio: float = 0.5,
                 init_std: float = 0.02,
                 init_scale: float = 1.0,
                 with_cls_token: bool = True,
                 **kwargs) -> None:
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)
        self.init_std = init_std
        self.init_scale = init_scale
        self.dropout_ratio = dropout_ratio
        self.with_cls_token = with_cls_token
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)

    def init_weights(self) -> None:
        """Initiate the parameters from scratch."""
        trunc_normal_init(self.fc_cls.weight, std=self.init_std)
        constant_init(self.fc_cls.bias, 0.02)
        self.fc_cls.weight.data.mul_(self.init_scale)
        self.fc_cls.bias.data.mul_(self.init_scale)

    def pre_logits(self, feats: Tuple[List[Tensor]]) -> Tensor:
        """The process before the final classification head.

        The input ``feats`` is a tuple of list of tensor, and each tensor is
        the feature of a backbone stage.
        """
        if self.with_cls_token:
            _, cls_token = feats[-1]
            return cls_token
        else:
            patch_token = feats[-1]
            return patch_token.mean(dim=(2, 3, 4))

    def forward(self, x: Tuple[List[Tensor]], **kwargs) -> Tensor:
        """Defines the computation performed at every call.

        Args:
            x (Tuple[List[Tensor]]): The input data.

        Returns:
            Tensor: The classification scores for input samples.
        """
        x = self.pre_logits(x)
        if self.dropout is not None:
            x = self.dropout(x)
        # [N, in_channels]
        cls_score = self.fc_cls(x)
        # [N, num_classes]
        return cls_score
