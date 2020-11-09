import math

import torch
from mmcv.cnn import ConvModule, constant_init, normal_init
from mmcv.runner import load_checkpoint
from mmcv.utils import _BatchNorm
from torch import nn as nn
from torch.utils import checkpoint as cp

from ...utils import get_root_logger
from ..registry import BACKBONES


class DLABasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 style='pytorch',
                 conv_cfg=dict(type='Conv'),
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True),
                 with_cp=False,
                 **kwargs):
        super().__init__()

        assert style in ['pytorch', 'caffe']

        self.conv1 = ConvModule(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv2 = ConvModule(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)
        self.relu = nn.ReLU(inplace=True)
        self.style = style
        self.stride = stride
        self.dilation = dilation
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.conv2(out)

        out = out + residual
        out = self.relu(out)

        return out


class DLABottleneck(nn.Module):

    expansion = 2

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 style='pytorch',
                 cardinality=1,
                 base_width=64,
                 conv_cfg=dict(type='Conv'),
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True),
                 with_cp=False):
        super().__init__()
        assert style in ['pytorch', 'caffe']

        mid_planes = int(math.floor(planes * (base_width / 64)) * cardinality)
        mid_planes = mid_planes // self.expansion

        if style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.conv1 = ConvModule(
            inplanes,
            mid_planes,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv2 = ConvModule(
            mid_planes,
            mid_planes,
            kernel_size=3,
            stride=self.conv2_stride,
            padding=dilation,
            dilation=dilation,
            groups=cardinality,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv3 = ConvModule(
            mid_planes,
            planes,
            kernel_size=1,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.dilation = dilation
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp

    def forward(self, x, residual=None):

        def _inner_forward(x, residual):
            """Forward wrapper for utilizing checkpoint."""

            if residual is None:
                residual = x

            out = self.conv1(x)
            out = self.conv2(out)
            out = self.conv3(out)

            out = out + residual

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x, residual)
        else:
            out = _inner_forward(x, residual)

        out = self.relu(out)

        return out


class DLARoot(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            residual,
            conv_cfg=dict(type='Conv'),
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=dict(type='ReLU', inplace=True),
    ):
        super().__init__()

        self.conv = ConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            padding=(kernel_size - 1) // 2,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        if self.residual:
            x = x + children[0]
        x = self.relu(x)

        return x


class DLATree(nn.Module):

    def __init__(self,
                 levels,
                 block,
                 in_channels,
                 out_channels,
                 stride=1,
                 dilation=1,
                 cardinality=1,
                 base_width=64,
                 level_root=False,
                 root_dim=0,
                 root_kernel_size=1,
                 root_residual=False):
        super().__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        self.downsample = nn.MaxPool2d(
            stride, stride=stride) if stride > 1 else nn.Identity()
        self.project = nn.Identity()
        cargs = dict(
            dilation=dilation, cardinality=cardinality, base_width=base_width)
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride, **cargs)
            self.tree2 = block(out_channels, out_channels, 1, **cargs)
        else:
            cargs.update(
                dict(
                    root_kernel_size=root_kernel_size,
                    root_residual=root_residual))
            self.tree1 = DLATree(
                levels - 1,
                block,
                in_channels,
                out_channels,
                stride,
                root_dim=0,
                **cargs)
            self.tree2 = DLATree(
                levels - 1,
                block,
                out_channels,
                out_channels,
                root_dim=root_dim + out_channels,
                **cargs)
        if levels == 1:
            self.root = DLARoot(root_dim, out_channels, root_kernel_size,
                                root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.levels = levels
        if in_channels != out_channels:
            self.project = ConvModule(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                bias=False,
                conv_cfg=dict(type='Conv'),
                norm_cfg=dict(type='BN', requires_grad=True),
                act_cfg=None)

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x)
        residual = self.project(bottom)
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


@BACKBONES.register_module()
class DLA(nn.Module):
    arch_settings = {
        'DLABasicBlock': DLABasicBlock,
        'DLABottleneck': DLABottleneck
    }
    OUTPUT_STRIDE = 32

    def __init__(self,
                 levels,
                 channels,
                 pretrained=None,
                 num_classes=1000,
                 in_channels=3,
                 cardinality=1,
                 base_width=64,
                 block='DLABasicBlock',
                 residual_root=False,
                 conv_cfg=dict(type='Conv'),
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True)):
        super().__init__()
        if block not in self.arch_settings:
            raise KeyError(f'invalid key {block} in DLA')
        self.channels = channels
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.cardinality = cardinality
        self.base_width = base_width
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.base_layer = ConvModule(
            in_channels,
            channels[0],
            kernel_size=7,
            stride=1,
            padding=3,
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.level0 = self._make_conv_level(channels[0], channels[0],
                                            levels[0])
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], stride=2)
        cargs = dict(
            cardinality=cardinality,
            base_width=base_width,
            root_residual=residual_root)
        self.level2 = DLATree(
            levels[2],
            block,
            channels[1],
            channels[2],
            2,
            level_root=False,
            **cargs)
        self.level3 = DLATree(
            levels[3],
            block,
            channels[2],
            channels[3],
            2,
            level_root=True,
            **cargs)
        self.level4 = DLATree(
            levels[4],
            block,
            channels[3],
            channels[4],
            2,
            level_root=True,
            **cargs)
        self.level5 = DLATree(
            levels[5],
            block,
            channels[4],
            channels[5],
            2,
            level_root=True,
            **cargs)
        self.feature_info = [
            dict(num_chs=channels[0], reduction=1,
                 module='level0'),  # rare to have a meaningful stride 1 level
            dict(num_chs=channels[1], reduction=2, module='level1'),
            dict(num_chs=channels[2], reduction=4, module='level2'),
            dict(num_chs=channels[3], reduction=8, module='level3'),
            dict(num_chs=channels[4], reduction=16, module='level4'),
            dict(num_chs=channels[5], reduction=32, module='level5')
        ]

        self.logger = get_root_logger()

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.append(
                ConvModule(
                    inplanes,
                    planes,
                    kernel_size=3,
                    stride=stride if i == 0 else 1,
                    padding=dilation,
                    bias=False,
                    dilation=dilation,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
            inplanes = planes
        return nn.Sequential(*modules)

    def init_weights(self):
        if isinstance(self.pretrained, str):
            load_checkpoint(
                self, self.pretrained, strict=False, logger=self.logger)
        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    normal_init(m, std=math.sqrt(2. / n))
                elif isinstance(m, _BatchNorm):
                    constant_init(m, 1)

    def forward(self, x):
        x = self.base_layer(x)
        for i in range(6):
            dla_layer = getattr(self, f'level{i}')
            x = dla_layer(x)
        return x
