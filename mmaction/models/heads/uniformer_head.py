# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

from mmengine.fileio import load
from mmengine.logging import MMLogger
from mmengine.runner.checkpoint import _load_checkpoint_with_prefix
from torch import Tensor, nn

from mmaction.registry import MODELS
from mmaction.utils import ConfigType, get_str_type
from .base import BaseHead


@MODELS.register_module()
class UniFormerHead(BaseHead):
    """Classification head for UniFormer. supports loading pretrained
    Kinetics-710 checkpoint to fine-tuning on other Kinetics dataset.

    A pytorch implement of: `UniFormerV2: Spatiotemporal
    Learning by Arming Image ViTs with Video UniFormer
    <https://arxiv.org/abs/2211.09552>`

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict or ConfigDict): Config for building loss.
            Defaults to `dict(type='CrossEntropyLoss')`.
        dropout_ratio (float): Probability of dropout layer.
            Defaults to : 0.0.
        channel_map (str, optional): Channel map file to selecting
            channels from pretrained head with extra channels.
            Defaults to None.
        init_cfg (dict or ConfigDict, optional): Config to control the
           initialization. Defaults to
           ``[
            dict(type='TruncNormal', layer='Linear', std=0.01)
           ]``.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 loss_cls: ConfigType = dict(type='CrossEntropyLoss'),
                 dropout_ratio: float = 0.0,
                 channel_map: Optional[str] = None,
                 init_cfg: Optional[dict] = dict(
                     type='TruncNormal', layer='Linear', std=0.02),
                 **kwargs) -> None:
        super().__init__(
            num_classes, in_channels, loss_cls, init_cfg=init_cfg, **kwargs)
        self.channel_map = channel_map
        self.dropout_ratio = dropout_ratio

        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)

    def _select_channels(self, stact_dict):
        selected_channels = load(self.channel_map)
        for key in stact_dict:
            stact_dict[key] = stact_dict[key][selected_channels]

    def init_weights(self) -> None:
        """Initiate the parameters from scratch."""
        if get_str_type(self.init_cfg['type']) == 'Pretrained':
            assert self.channel_map is not None, \
                'load cls_head weights needs to specify the channel map file'
            logger = MMLogger.get_current_instance()
            pretrained = self.init_cfg['checkpoint']
            logger.info(f'load pretrained model from {pretrained}')
            state_dict = _load_checkpoint_with_prefix(
                'cls_head.', pretrained, map_location='cpu')
            self._select_channels(state_dict)
            msg = self.load_state_dict(state_dict, strict=False)
            logger.info(msg)
        else:
            super().init_weights()

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        """Defines the computation performed at every call.

        Args:
            x (Tensor): The input data.

        Returns:
            Tensor: The classification scores for input samples.
        """
        # [N, in_channels]
        if self.dropout is not None:
            x = self.dropout(x)
        # [N, in_channels]
        cls_score = self.fc_cls(x)
        # [N, num_classes]
        return cls_score
