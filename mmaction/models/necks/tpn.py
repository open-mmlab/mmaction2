# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.model.weight_init import constant_init, normal_init, xavier_init
from torch import Tensor

from mmaction.registry import MODELS
from mmaction.utils import ConfigType, OptConfigType, SampleList


class DownSample(nn.Module):
    """DownSample modules.

    It uses convolution and maxpooling to downsample the input feature,
    and specifies downsample position to determine `pool-conv` or `conv-pool`.

    Args:
        in_channels (int): Channel number of input features.
        out_channels (int): Channel number of output feature.
        kernel_size (int or Tuple[int]): Same as :class:`ConvModule`.
            Defaults to ``(3, 1, 1)``.
        stride (int or Tuple[int]): Same as :class:`ConvModule`.
            Defaults to ``(1, 1, 1)``.
        padding (int or Tuple[int]): Same as :class:`ConvModule`.
            Defaults to ``(1, 0, 0)``.
        groups (int): Same as :class:`ConvModule`. Defaults to 1.
        bias (bool or str): Same as :class:`ConvModule`. Defaults to False.
        conv_cfg (dict or ConfigDict): Same as :class:`ConvModule`.
            Defaults to ``dict(type='Conv3d')``.
        norm_cfg (dict or ConfigDict, optional): Same as :class:`ConvModule`.
            Defaults to None.
        act_cfg (dict or ConfigDict, optional): Same as :class:`ConvModule`.
            Defaults to None.
        downsample_position (str): Type of downsample position. Options are
            ``before`` and ``after``. Defaults to ``after``.
        downsample_scale (int or Tuple[int]): downsample scale for maxpooling.
            It will be used for kernel size and stride of maxpooling.
            Defaults to ``(1, 2, 2)``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int]] = (3, 1, 1),
        stride: Union[int, Tuple[int]] = (1, 1, 1),
        padding: Union[int, Tuple[int]] = (1, 0, 0),
        groups: int = 1,
        bias: Union[bool, str] = False,
        conv_cfg: ConfigType = dict(type='Conv3d'),
        norm_cfg: OptConfigType = None,
        act_cfg: OptConfigType = None,
        downsample_position: str = 'after',
        downsample_scale: Union[int, Tuple[int]] = (1, 2, 2)
    ) -> None:
        super().__init__()
        self.conv = ConvModule(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=bias,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        assert downsample_position in ['before', 'after']
        self.downsample_position = downsample_position
        self.pool = nn.MaxPool3d(
            downsample_scale, downsample_scale, (0, 0, 0), ceil_mode=True)

    def forward(self, x: Tensor) -> Tensor:
        if self.downsample_position == 'before':
            x = self.pool(x)
            x = self.conv(x)
        else:
            x = self.conv(x)
            x = self.pool(x)
        return x


class LevelFusion(nn.Module):
    """Level Fusion module.

    This module is used to aggregate the hierarchical features dynamic in
    visual tempos and consistent in spatial semantics. The top/bottom features
    for top-down/bottom-up flow would be combined to achieve two additional
    options, namely 'Cascade Flow' or 'Parallel Flow'. While applying a
    bottom-up flow after a top-down flow will lead to the cascade flow,
    applying them simultaneously will result in the parallel flow.

    Args:
        in_channels (Tuple[int]): Channel numbers of input features tuple.
        mid_channels (Tuple[int]): Channel numbers of middle features tuple.
        out_channels (int): Channel numbers of output features.
        downsample_scales (Tuple[int | Tuple[int]]): downsample scales for
            each :class:`DownSample` module.
            Defaults to ``((1, 1, 1), (1, 1, 1))``.
    """

    def __init__(
        self,
        in_channels: Tuple[int],
        mid_channels: Tuple[int],
        out_channels: int,
        downsample_scales: Tuple[int, Tuple[int]] = ((1, 1, 1), (1, 1, 1))
    ) -> None:
        super().__init__()
        num_stages = len(in_channels)

        self.downsamples = nn.ModuleList()
        for i in range(num_stages):
            downsample = DownSample(
                in_channels[i],
                mid_channels[i],
                kernel_size=(1, 1, 1),
                stride=(1, 1, 1),
                bias=False,
                padding=(0, 0, 0),
                groups=32,
                norm_cfg=dict(type='BN3d', requires_grad=True),
                act_cfg=dict(type='ReLU', inplace=True),
                downsample_position='before',
                downsample_scale=downsample_scales[i])
            self.downsamples.append(downsample)

        self.fusion_conv = ConvModule(
            sum(mid_channels),
            out_channels,
            1,
            stride=1,
            padding=0,
            bias=False,
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=dict(type='BN3d', requires_grad=True),
            act_cfg=dict(type='ReLU', inplace=True))

    def forward(self, x: Tuple[Tensor]) -> Tensor:
        out = [self.downsamples[i](feature) for i, feature in enumerate(x)]
        out = torch.cat(out, 1)
        out = self.fusion_conv(out)

        return out


class SpatialModulation(nn.Module):
    """Spatial Semantic Modulation.

    This module is used to align spatial semantics of features in the
    multi-depth pyramid. For each but the top-level feature, a stack
    of convolutions with level-specific stride are applied to it, matching
    its spatial shape and receptive field with the top one.

    Args:
        in_channels (Tuple[int]): Channel numbers of input features tuple.
        out_channels (int): Channel numbers of output features tuple.
    """

    def __init__(self, in_channels: Tuple[int], out_channels: int) -> None:
        super().__init__()

        self.spatial_modulation = nn.ModuleList()
        for channel in in_channels:
            downsample_scale = out_channels // channel
            downsample_factor = int(np.log2(downsample_scale))
            op = nn.ModuleList()
            if downsample_factor < 1:
                op = nn.Identity()
            else:
                for factor in range(downsample_factor):
                    in_factor = 2**factor
                    out_factor = 2**(factor + 1)
                    op.append(
                        ConvModule(
                            channel * in_factor,
                            channel * out_factor, (1, 3, 3),
                            stride=(1, 2, 2),
                            padding=(0, 1, 1),
                            bias=False,
                            conv_cfg=dict(type='Conv3d'),
                            norm_cfg=dict(type='BN3d', requires_grad=True),
                            act_cfg=dict(type='ReLU', inplace=True)))
            self.spatial_modulation.append(op)

    def forward(self, x: Tuple[Tensor]) -> list:
        out = []
        for i, _ in enumerate(x):
            if isinstance(self.spatial_modulation[i], nn.ModuleList):
                out_ = x[i]
                for op in self.spatial_modulation[i]:
                    out_ = op(out_)
                out.append(out_)
            else:
                out.append(self.spatial_modulation[i](x[i]))
        return out


class AuxHead(nn.Module):
    """Auxiliary Head.

    This auxiliary head is appended to receive stronger supervision,
    leading to enhanced semantics.

    Args:
        in_channels (int): Channel number of input features.
        out_channels (int): Channel number of output features.
        loss_weight (float): weight of loss for the auxiliary head.
            Defaults to 0.5.
        loss_cls (dict or ConfigDict): Config for building loss.
            Defaults to ``dict(type='CrossEntropyLoss')``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        loss_weight: float = 0.5,
        loss_cls: ConfigType = dict(type='CrossEntropyLoss')
    ) -> None:
        super().__init__()

        self.conv = ConvModule(
            in_channels,
            in_channels * 2, (1, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
            bias=False,
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=dict(type='BN3d', requires_grad=True))
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.loss_weight = loss_weight
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(in_channels * 2, out_channels)
        self.loss_cls = MODELS.build(loss_cls)

    def init_weights(self) -> None:
        """Initiate the parameters from scratch."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                normal_init(m, std=0.01)
            if isinstance(m, nn.Conv3d):
                xavier_init(m, distribution='uniform')
            if isinstance(m, nn.BatchNorm3d):
                constant_init(m, 1)

    def loss(self, x: Tensor, data_samples: Optional[SampleList]) -> dict:
        """Calculate auxiliary loss."""
        x = self(x)
        labels = [x.gt_labels.item for x in data_samples]
        labels = torch.stack(labels).to(x.device)
        labels = labels.squeeze()
        if labels.shape == torch.Size([]):
            labels = labels.unsqueeze(0)

        losses = dict()
        losses['loss_aux'] = self.loss_weight * self.loss_cls(x, labels)
        return losses

    def forward(self, x: Tensor) -> Tensor:
        """Auxiliary head forward function."""
        x = self.conv(x)
        x = self.avg_pool(x).squeeze(-1).squeeze(-1).squeeze(-1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


class TemporalModulation(nn.Module):
    """Temporal Rate Modulation.

    The module is used to equip TPN with a similar flexibility for temporal
    tempo modulation as in the input-level frame pyramid.

    Args:
        in_channels (int): Channel number of input features.
        out_channels (int): Channel number of output features.
        downsample_scale (int): Downsample scale for maxpooling. Defaults to 8.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 downsample_scale: int = 8) -> None:
        super().__init__()

        self.conv = ConvModule(
            in_channels,
            out_channels, (3, 1, 1),
            stride=(1, 1, 1),
            padding=(1, 0, 0),
            bias=False,
            groups=32,
            conv_cfg=dict(type='Conv3d'),
            act_cfg=None)
        self.pool = nn.MaxPool3d((downsample_scale, 1, 1),
                                 (downsample_scale, 1, 1), (0, 0, 0),
                                 ceil_mode=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.pool(x)
        return x


@MODELS.register_module()
class TPN(nn.Module):
    """TPN neck.

    This module is proposed in `Temporal Pyramid Network for Action Recognition
    <https://arxiv.org/pdf/2004.03548.pdf>`_

    Args:
        in_channels (Tuple[int]): Channel numbers of input features tuple.
        out_channels (int): Channel number of output feature.
        spatial_modulation_cfg (dict or ConfigDict, optional): Config for
            spatial modulation layers. Required keys are ``in_channels`` and
            ``out_channels``. Defaults to None.
        temporal_modulation_cfg (dict or ConfigDict, optional): Config for
            temporal modulation layers. Defaults to None.
        upsample_cfg (dict or ConfigDict, optional): Config for upsample
            layers. The keys are same as that in :class:``nn.Upsample``.
            Defaults to None.
        downsample_cfg (dict or ConfigDict, optional): Config for downsample
            layers. Defaults to None.
        level_fusion_cfg (dict or ConfigDict, optional): Config for level
            fusion layers.
            Required keys are ``in_channels``, ``mid_channels``,
            ``out_channels``. Defaults to None.
        aux_head_cfg (dict or ConfigDict, optional): Config for aux head
            layers. Required keys are ``out_channels``. Defaults to None.
        flow_type (str): Flow type to combine the features. Options are
            ``cascade`` and ``parallel``. Defaults to ``cascade``.
    """

    def __init__(self,
                 in_channels: Tuple[int],
                 out_channels: int,
                 spatial_modulation_cfg: OptConfigType = None,
                 temporal_modulation_cfg: OptConfigType = None,
                 upsample_cfg: OptConfigType = None,
                 downsample_cfg: OptConfigType = None,
                 level_fusion_cfg: OptConfigType = None,
                 aux_head_cfg: OptConfigType = None,
                 flow_type: str = 'cascade') -> None:
        super().__init__()
        assert isinstance(in_channels, tuple)
        assert isinstance(out_channels, int)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_tpn_stages = len(in_channels)

        assert spatial_modulation_cfg is None or isinstance(
            spatial_modulation_cfg, dict)
        assert temporal_modulation_cfg is None or isinstance(
            temporal_modulation_cfg, dict)
        assert upsample_cfg is None or isinstance(upsample_cfg, dict)
        assert downsample_cfg is None or isinstance(downsample_cfg, dict)
        assert aux_head_cfg is None or isinstance(aux_head_cfg, dict)
        assert level_fusion_cfg is None or isinstance(level_fusion_cfg, dict)

        if flow_type not in ['cascade', 'parallel']:
            raise ValueError(
                f"flow type in TPN should be 'cascade' or 'parallel', "
                f'but got {flow_type} instead.')
        self.flow_type = flow_type

        self.temporal_modulation_ops = nn.ModuleList()
        self.upsample_ops = nn.ModuleList()
        self.downsample_ops = nn.ModuleList()

        self.level_fusion_1 = LevelFusion(**level_fusion_cfg)
        self.spatial_modulation = SpatialModulation(**spatial_modulation_cfg)

        for i in range(self.num_tpn_stages):

            if temporal_modulation_cfg is not None:
                downsample_scale = temporal_modulation_cfg[
                    'downsample_scales'][i]
                temporal_modulation = TemporalModulation(
                    in_channels[-1], out_channels, downsample_scale)
                self.temporal_modulation_ops.append(temporal_modulation)

            if i < self.num_tpn_stages - 1:
                if upsample_cfg is not None:
                    upsample = nn.Upsample(**upsample_cfg)
                    self.upsample_ops.append(upsample)

                if downsample_cfg is not None:
                    downsample = DownSample(out_channels, out_channels,
                                            **downsample_cfg)
                    self.downsample_ops.append(downsample)

        out_dims = level_fusion_cfg['out_channels']

        # two pyramids
        self.level_fusion_2 = LevelFusion(**level_fusion_cfg)

        self.pyramid_fusion = ConvModule(
            out_dims * 2,
            2048,
            1,
            stride=1,
            padding=0,
            bias=False,
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=dict(type='BN3d', requires_grad=True))

        if aux_head_cfg is not None:
            self.aux_head = AuxHead(self.in_channels[-2], **aux_head_cfg)
        else:
            self.aux_head = None

    def init_weights(self) -> None:
        """Default init_weights for conv(msra) and norm in ConvModule."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                xavier_init(m, distribution='uniform')
            if isinstance(m, nn.BatchNorm3d):
                constant_init(m, 1)

        if self.aux_head is not None:
            self.aux_head.init_weights()

    def forward(self,
                x: Tuple[Tensor],
                data_samples: Optional[SampleList] = None) -> tuple:

        loss_aux = dict()
        # Calculate auxiliary loss if `self.aux_head`
        # and `data_samples` are not None.
        if self.aux_head is not None and data_samples is not None:
            loss_aux = self.aux_head.loss(x[-2], data_samples)

        # Spatial Modulation
        spatial_modulation_outs = self.spatial_modulation(x)

        # Temporal Modulation
        temporal_modulation_outs = []
        for i, temporal_modulation in enumerate(self.temporal_modulation_ops):
            temporal_modulation_outs.append(
                temporal_modulation(spatial_modulation_outs[i]))

        outs = [out.clone() for out in temporal_modulation_outs]
        if len(self.upsample_ops) != 0:
            for i in range(self.num_tpn_stages - 1, 0, -1):
                outs[i - 1] = outs[i - 1] + self.upsample_ops[i - 1](outs[i])

        # Get top-down outs
        top_down_outs = self.level_fusion_1(outs)

        # Build bottom-up flow using downsample operation
        if self.flow_type == 'parallel':
            outs = [out.clone() for out in temporal_modulation_outs]
        if len(self.downsample_ops) != 0:
            for i in range(self.num_tpn_stages - 1):
                outs[i + 1] = outs[i + 1] + self.downsample_ops[i](outs[i])

        # Get bottom-up outs
        botton_up_outs = self.level_fusion_2(outs)

        # fuse two pyramid outs
        outs = self.pyramid_fusion(
            torch.cat([top_down_outs, botton_up_outs], 1))

        return outs, loss_aux
