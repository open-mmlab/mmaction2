# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import ConvModule, Swish, build_activation_layer
from mmengine.logging import MMLogger
from mmengine.model.weight_init import constant_init, kaiming_init
from mmengine.runner import load_checkpoint
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm

from mmaction.registry import MODELS


class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.bottleneck = self._round_width(channels, reduction)
        self.fc1 = nn.Conv3d(
            channels, self.bottleneck, kernel_size=1, padding=0)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv3d(
            self.bottleneck, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def _round_width(width, multiplier, min_width=8, divisor=8):
        """Round width of filters based on width multiplier."""
        width *= multiplier
        min_width = min_width or divisor
        width_out = max(min_width,
                        int(width + divisor / 2) // divisor * divisor)
        if width_out < 0.9 * width:
            width_out += divisor
        return int(width_out)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (Tensor): The input data.

        Returns:
            Tensor: The output of the module.
        """
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class BlockX3D(nn.Module):
    """BlockX3D 3d building block for X3D.

    Args:
        inplanes (int): Number of channels for the input in first conv3d layer.
        planes (int): Number of channels produced by some norm/conv3d layers.
        outplanes (int): Number of channels produced by final the conv3d layer.
        spatial_stride (int): Spatial stride in the conv3d layer. Default: 1.
        downsample (nn.Module | None): Downsample layer. Default: None.
        se_ratio (float | None): The reduction ratio of squeeze and excitation
            unit. If set as None, it means not using SE unit. Default: None.
        use_swish (bool): Whether to use swish as the activation function
            before and after the 3x3x3 conv. Default: True.
        conv_cfg (dict): Config dict for convolution layer.
            Default: ``dict(type='Conv3d')``.
        norm_cfg (dict): Config for norm layers. required keys are ``type``,
            Default: ``dict(type='BN3d')``.
        act_cfg (dict): Config dict for activation layer.
            Default: ``dict(type='ReLU')``.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
    """

    def __init__(self,
                 inplanes,
                 planes,
                 outplanes,
                 spatial_stride=1,
                 downsample=None,
                 se_ratio=None,
                 use_swish=True,
                 conv_cfg=dict(type='Conv3d'),
                 norm_cfg=dict(type='BN3d'),
                 act_cfg=dict(type='ReLU'),
                 with_cp=False):
        super().__init__()

        self.inplanes = inplanes
        self.planes = planes
        self.outplanes = outplanes
        self.spatial_stride = spatial_stride
        self.downsample = downsample
        self.se_ratio = se_ratio
        self.use_swish = use_swish
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.act_cfg_swish = dict(type='Swish')
        self.with_cp = with_cp

        self.conv1 = ConvModule(
            in_channels=inplanes,
            out_channels=planes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        # Here we use the channel-wise conv
        self.conv2 = ConvModule(
            in_channels=planes,
            out_channels=planes,
            kernel_size=3,
            stride=(1, self.spatial_stride, self.spatial_stride),
            padding=1,
            groups=planes,
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=None)

        self.swish = Swish()

        self.conv3 = ConvModule(
            in_channels=planes,
            out_channels=outplanes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=None)

        if self.se_ratio is not None:
            self.se_module = SEModule(planes, self.se_ratio)

        self.relu = build_activation_layer(self.act_cfg)

    def forward(self, x):
        """Defines the computation performed at every call."""

        def _inner_forward(x):
            """Forward wrapper for utilizing checkpoint."""
            identity = x

            out = self.conv1(x)
            out = self.conv2(out)
            if self.se_ratio is not None:
                out = self.se_module(out)

            out = self.swish(out)

            out = self.conv3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out = out + identity
            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)
        out = self.relu(out)
        return out


# We do not support initialize with 2D pretrain weight for X3D
@MODELS.register_module()
class X3D(nn.Module):
    """X3D backbone. https://arxiv.org/pdf/2004.04730.pdf.

    Args:
        gamma_w (float): Global channel width expansion factor. Default: 1.
        gamma_b (float): Bottleneck channel width expansion factor. Default: 1.
        gamma_d (float): Network depth expansion factor. Default: 1.
        pretrained (str | None): Name of pretrained model. Default: None.
        in_channels (int): Channel num of input features. Default: 3.
        num_stages (int): Resnet stages. Default: 4.
        spatial_strides (Sequence[int]):
            Spatial strides of residual blocks of each stage.
            Default: ``(1, 2, 2, 2)``.
        frozen_stages (int): Stages to be frozen (all param fixed). If set to
            -1, it means not freezing any parameters. Default: -1.
        se_style (str): The style of inserting SE modules into BlockX3D, 'half'
            denotes insert into half of the blocks, while 'all' denotes insert
            into all blocks. Default: 'half'.
        se_ratio (float | None): The reduction ratio of squeeze and excitation
            unit. If set as None, it means not using SE unit. Default: 1 / 16.
        use_swish (bool): Whether to use swish as the activation function
            before and after the 3x3x3 conv. Default: True.
        conv_cfg (dict): Config for conv layers. required keys are ``type``
            Default: ``dict(type='Conv3d')``.
        norm_cfg (dict): Config for norm layers. required keys are ``type`` and
            ``requires_grad``.
            Default: ``dict(type='BN3d', requires_grad=True)``.
        act_cfg (dict): Config dict for activation layer.
            Default: ``dict(type='ReLU', inplace=True)``.
        norm_eval (bool): Whether to set BN layers to eval mode, namely, freeze
            running stats (mean and var). Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        zero_init_residual (bool):
            Whether to use zero initialization for residual block,
            Default: True.
        kwargs (dict, optional): Key arguments for "make_res_layer".
    """

    def __init__(self,
                 gamma_w=1.0,
                 gamma_b=1.0,
                 gamma_d=1.0,
                 pretrained=None,
                 in_channels=3,
                 num_stages=4,
                 spatial_strides=(2, 2, 2, 2),
                 frozen_stages=-1,
                 se_style='half',
                 se_ratio=1 / 16,
                 use_swish=True,
                 conv_cfg=dict(type='Conv3d'),
                 norm_cfg=dict(type='BN3d', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_eval=False,
                 with_cp=False,
                 zero_init_residual=True,
                 **kwargs):
        super().__init__()
        self.gamma_w = gamma_w
        self.gamma_b = gamma_b
        self.gamma_d = gamma_d

        self.pretrained = pretrained
        self.in_channels = in_channels
        # Hard coded, can be changed by gamma_w
        self.base_channels = 24
        self.stage_blocks = [1, 2, 5, 3]

        # apply parameters gamma_w and gamma_d
        self.base_channels = self._round_width(self.base_channels,
                                               self.gamma_w)

        self.stage_blocks = [
            self._round_repeats(x, self.gamma_d) for x in self.stage_blocks
        ]

        self.num_stages = num_stages
        assert 1 <= num_stages <= 4
        self.spatial_strides = spatial_strides
        assert len(spatial_strides) == num_stages
        self.frozen_stages = frozen_stages

        self.se_style = se_style
        assert self.se_style in ['all', 'half']
        self.se_ratio = se_ratio
        assert (self.se_ratio is None) or (self.se_ratio > 0)
        self.use_swish = use_swish

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.zero_init_residual = zero_init_residual

        self.block = BlockX3D
        self.stage_blocks = self.stage_blocks[:num_stages]
        self.layer_inplanes = self.base_channels
        self._make_stem_layer()

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            spatial_stride = spatial_strides[i]
            inplanes = self.base_channels * 2**i
            planes = int(inplanes * self.gamma_b)

            res_layer = self.make_res_layer(
                self.block,
                self.layer_inplanes,
                inplanes,
                planes,
                num_blocks,
                spatial_stride=spatial_stride,
                se_style=self.se_style,
                se_ratio=self.se_ratio,
                use_swish=self.use_swish,
                norm_cfg=self.norm_cfg,
                conv_cfg=self.conv_cfg,
                act_cfg=self.act_cfg,
                with_cp=with_cp,
                **kwargs)
            self.layer_inplanes = inplanes
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self.feat_dim = self.base_channels * 2**(len(self.stage_blocks) - 1)
        self.conv5 = ConvModule(
            self.feat_dim,
            int(self.feat_dim * self.gamma_b),
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.feat_dim = int(self.feat_dim * self.gamma_b)

    @staticmethod
    def _round_width(width, multiplier, min_depth=8, divisor=8):
        """Round width of filters based on width multiplier."""
        if not multiplier:
            return width

        width *= multiplier
        min_depth = min_depth or divisor
        new_filters = max(min_depth,
                          int(width + divisor / 2) // divisor * divisor)
        if new_filters < 0.9 * width:
            new_filters += divisor
        return int(new_filters)

    @staticmethod
    def _round_repeats(repeats, multiplier):
        """Round number of layers based on depth multiplier."""
        if not multiplier:
            return repeats
        return int(math.ceil(multiplier * repeats))

    # the module is parameterized with gamma_b
    # no temporal_stride
    def make_res_layer(self,
                       block,
                       layer_inplanes,
                       inplanes,
                       planes,
                       blocks,
                       spatial_stride=1,
                       se_style='half',
                       se_ratio=None,
                       use_swish=True,
                       norm_cfg=None,
                       act_cfg=None,
                       conv_cfg=None,
                       with_cp=False,
                       **kwargs):
        """Build residual layer for ResNet3D.

        Args:
            block (nn.Module): Residual module to be built.
            layer_inplanes (int): Number of channels for the input feature
                of the res layer.
            inplanes (int): Number of channels for the input feature in each
                block, which equals to base_channels * gamma_w.
            planes (int): Number of channels for the output feature in each
                block, which equals to base_channel * gamma_w * gamma_b.
            blocks (int): Number of residual blocks.
            spatial_stride (int): Spatial strides in residual and conv layers.
                Default: 1.
            se_style (str): The style of inserting SE modules into BlockX3D,
                'half' denotes insert into half of the blocks, while 'all'
                denotes insert into all blocks. Default: 'half'.
            se_ratio (float | None): The reduction ratio of squeeze and
                excitation unit. If set as None, it means not using SE unit.
                Default: None.
            use_swish (bool): Whether to use swish as the activation function
                before and after the 3x3x3 conv. Default: True.
            conv_cfg (dict | None): Config for norm layers. Default: None.
            norm_cfg (dict | None): Config for norm layers. Default: None.
            act_cfg (dict | None): Config for activate layers. Default: None.
            with_cp (bool | None): Use checkpoint or not. Using checkpoint
                will save some memory while slowing down the training speed.
                Default: False.

        Returns:
            nn.Module: A residual layer for the given config.
        """
        downsample = None
        if spatial_stride != 1 or layer_inplanes != inplanes:
            downsample = ConvModule(
                layer_inplanes,
                inplanes,
                kernel_size=1,
                stride=(1, spatial_stride, spatial_stride),
                padding=0,
                bias=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None)

        use_se = [False] * blocks
        if self.se_style == 'all':
            use_se = [True] * blocks
        elif self.se_style == 'half':
            use_se = [i % 2 == 0 for i in range(blocks)]
        else:
            raise NotImplementedError

        layers = []
        layers.append(
            block(
                layer_inplanes,
                planes,
                inplanes,
                spatial_stride=spatial_stride,
                downsample=downsample,
                se_ratio=se_ratio if use_se[0] else None,
                use_swish=use_swish,
                norm_cfg=norm_cfg,
                conv_cfg=conv_cfg,
                act_cfg=act_cfg,
                with_cp=with_cp,
                **kwargs))

        for i in range(1, blocks):
            layers.append(
                block(
                    inplanes,
                    planes,
                    inplanes,
                    spatial_stride=1,
                    se_ratio=se_ratio if use_se[i] else None,
                    use_swish=use_swish,
                    norm_cfg=norm_cfg,
                    conv_cfg=conv_cfg,
                    act_cfg=act_cfg,
                    with_cp=with_cp,
                    **kwargs))

        return nn.Sequential(*layers)

    def _make_stem_layer(self):
        """Construct the stem layers consists of a conv+norm+act module and a
        pooling layer."""
        self.conv1_s = ConvModule(
            self.in_channels,
            self.base_channels,
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=None,
            act_cfg=None)
        self.conv1_t = ConvModule(
            self.base_channels,
            self.base_channels,
            kernel_size=(5, 1, 1),
            stride=(1, 1, 1),
            padding=(2, 0, 0),
            groups=self.base_channels,
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def _freeze_stages(self):
        """Prevent all the parameters from being optimized before
        ``self.frozen_stages``."""
        if self.frozen_stages >= 0:
            self.conv1_s.eval()
            self.conv1_t.eval()
            for param in self.conv1_s.parameters():
                param.requires_grad = False
            for param in self.conv1_t.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if isinstance(self.pretrained, str):
            logger = MMLogger.get_current_instance()
            logger.info(f'load model from: {self.pretrained}')

            load_checkpoint(self, self.pretrained, strict=False, logger=logger)

        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    kaiming_init(m)
                elif isinstance(m, _BatchNorm):
                    constant_init(m, 1)

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, BlockX3D):
                        constant_init(m.conv3.bn, 0)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The feature of the input
            samples extracted by the backbone.
        """
        x = self.conv1_s(x)
        x = self.conv1_t(x)
        for layer_name in self.res_layers:
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
        x = self.conv5(x)
        return x

    def train(self, mode=True):
        """Set the optimization status when training."""
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
