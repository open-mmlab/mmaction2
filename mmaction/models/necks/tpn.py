import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, constant_init, normal_init, xavier_init

from ..builder import NECKS, build_loss


class Identity(nn.Module):
    """Identity mapping."""

    def forward(self, x):
        return x


class DownSample(nn.Module):
    """DownSample modules.

    It uses convolution and maxpooling to downsample the input feature,
    and specifies downsample position to determine `pool-conv` or `conv-pool`.

    Args:
        in_channels (int): Channel number of input features.
        out_channels (int): Channel number of output feature.
        kernel_size (int | tuple[int]): Same as :class:`ConvModule`.
            Default: (3, 1, 1).
        stride (int | tuple[int]): Same as :class:`ConvModule`.
            Default: (1, 1, 1).
        padding (int | tuple[int]): Same as :class:`ConvModule`.
            Default: (1, 0, 0).
        groups (int): Same as :class:`ConvModule`. Default: 1.
        bias (bool | str): Same as :class:`ConvModule`. Default: False.
        conv_cfg (dict | None): Same as :class:`ConvModule`.
            Default: dict(type='Conv3d').
        norm_cfg (dict | None): Same as :class:`ConvModule`. Default: None.
        act_cfg (dict | None): Same as :class:`ConvModule`. Default: None.
        downsample_position (str): Type of downsample position. Options are
            'before' and 'after'. Default: 'after'.
        downsample_scale (int | tuple[int]): downsample scale for maxpooling.
            It will be used for kernel size and stride of maxpooling.
            Default: (1, 2, 2).
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=(3, 1, 1),
                 stride=(1, 1, 1),
                 padding=(1, 0, 0),
                 groups=1,
                 bias=False,
                 conv_cfg=dict(type='Conv3d'),
                 norm_cfg=None,
                 act_cfg=None,
                 downsample_position='after',
                 downsample_scale=(1, 2, 2)):
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

    def forward(self, x):
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
        in_channels (tuple[int]): Channel numbers of input features tuple.
        mid_channels (tuple[int]): Channel numbers of middle features tuple.
        out_channels (int): Channel numbers of output features.
        downsample_scales (tuple[int | tuple[int]]): downsample scales for
            each :class:`DownSample` module. Default: ((1, 1, 1), (1, 1, 1)).
    """

    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 downsample_scales=((1, 1, 1), (1, 1, 1))):
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

    def forward(self, x):
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
        in_channels (tuple[int]): Channel numbers of input features tuple.
        out_channels (int): Channel numbers of output features tuple.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.spatial_modulation = nn.ModuleList()
        for channel in in_channels:
            downsample_scale = out_channels // channel
            downsample_factor = int(np.log2(downsample_scale))
            op = nn.ModuleList()
            if downsample_factor < 1:
                op = Identity()
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

    def forward(self, x):
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
            Default: 0.5.
        loss_cls (dict): loss_cls (dict): Config for building loss.
            Default: ``dict(type='CrossEntropyLoss')``.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 loss_weight=0.5,
                 loss_cls=dict(type='CrossEntropyLoss')):
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
        self.loss_cls = build_loss(loss_cls)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                normal_init(m, std=0.01)
            if isinstance(m, nn.Conv3d):
                xavier_init(m, distribution='uniform')
            if isinstance(m, nn.BatchNorm3d):
                constant_init(m, 1)

    def forward(self, x, target=None):
        losses = dict()
        if target is None:
            return losses
        x = self.conv(x)
        x = self.avg_pool(x).squeeze(-1).squeeze(-1).squeeze(-1)
        x = self.dropout(x)
        x = self.fc(x)

        if target.shape == torch.Size([]):
            target = target.unsqueeze(0)

        losses['loss_aux'] = self.loss_weight * self.loss_cls(x, target)
        return losses


class TemporalModulation(nn.Module):
    """Temporal Rate Modulation.

    The module is used to equip TPN with a similar flexibility for temporal
    tempo modulation as in the input-level frame pyramid.

    Args:
        in_channels (int): Channel number of input features.
        out_channels (int): Channel number of output features.
        downsample_scale (int): Downsample scale for maxpooling. Default: 8.
    """

    def __init__(self, in_channels, out_channels, downsample_scale=8):
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

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x


@NECKS.register_module()
class TPN(nn.Module):
    """TPN neck.

    This module is proposed in `Temporal Pyramid Network for Action Recognition
    <https://arxiv.org/pdf/2004.03548.pdf>`_

    Args:
        in_channels (tuple[int]): Channel numbers of input features tuple.
        out_channels (int): Channel number of output feature.
        spatial_modulation_cfg (dict | None): Config for spatial modulation
            layers. Required keys are `in_channels` and `out_channels`.
            Default: None.
        temporal_modulation_cfg (dict | None): Config for temporal modulation
            layers. Default: None.
        upsample_cfg (dict | None): Config for upsample layers. The keys are
            same as that in :class:``nn.Upsample``. Default: None.
        downsample_cfg (dict | None): Config for downsample layers.
            Default: None.
        level_fusion_cfg (dict | None): Config for level fusion layers.
            Required keys are 'in_channels', 'mid_channels', 'out_channels'.
            Default: None.
        aux_head_cfg (dict | None): Config for aux head layers.
            Required keys are 'out_channels'. Default: None.
        flow_type (str): Flow type to combine the features. Options are
            'cascade' and 'parallel'. Default: 'cascade'.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 spatial_modulation_cfg=None,
                 temporal_modulation_cfg=None,
                 upsample_cfg=None,
                 downsample_cfg=None,
                 level_fusion_cfg=None,
                 aux_head_cfg=None,
                 flow_type='cascade'):
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
        self.init_weights()

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                xavier_init(m, distribution='uniform')
            if isinstance(m, nn.BatchNorm3d):
                constant_init(m, 1)

        if self.aux_head is not None:
            self.aux_head.init_weights()

    def forward(self, x, target=None):
        loss_aux = dict()

        # Auxiliary loss
        if self.aux_head is not None:
            loss_aux = self.aux_head(x[-2], target)

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
