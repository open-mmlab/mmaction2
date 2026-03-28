#!/usr/bin/env python

# this file is from https://github.com/ucbdrive/dla/blob/master/dla.py.

import math
from os.path import join

import fvcore.nn.weight_init as weight_init
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from detectron2.layers import Conv2d, ModulatedDeformConv, ShapeSpec
from detectron2.layers.batch_norm import get_norm
from detectron2.modeling.backbone import FPN, Backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from torch import nn

WEB_ROOT = 'http://dl.yf.io/dla/models'


def get_model_url(data, name, hash):
    return join('http://dl.yf.io/dla/models', data,
                '{}-{}.pth'.format(name, hash))


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


class BasicBlock(nn.Module):

    def __init__(self, cfg, inplanes, planes, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            bias=False,
            dilation=dilation)
        self.bn1 = get_norm(cfg.MODEL.DLA.NORM, planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=dilation,
            bias=False,
            dilation=dilation)
        self.bn2 = get_norm(cfg.MODEL.DLA.NORM, planes)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, cfg, inplanes, planes, stride=1, dilation=1):
        super(Bottleneck, self).__init__()
        expansion = Bottleneck.expansion
        bottle_planes = planes // expansion
        self.conv1 = nn.Conv2d(
            inplanes, bottle_planes, kernel_size=1, bias=False)
        self.bn1 = get_norm(cfg.MODEL.DLA.NORM, bottle_planes)
        self.conv2 = nn.Conv2d(
            bottle_planes,
            bottle_planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            bias=False,
            dilation=dilation)
        self.bn2 = get_norm(cfg.MODEL.DLA.NORM, bottle_planes)
        self.conv3 = nn.Conv2d(
            bottle_planes, planes, kernel_size=1, bias=False)
        self.bn3 = get_norm(cfg.MODEL.DLA.NORM, planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class Root(nn.Module):

    def __init__(self, cfg, in_channels, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            bias=False,
            padding=(kernel_size - 1) // 2)
        self.bn = get_norm(cfg.MODEL.DLA.NORM, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


class Tree(nn.Module):

    def __init__(self,
                 cfg,
                 levels,
                 block,
                 in_channels,
                 out_channels,
                 stride=1,
                 level_root=False,
                 root_dim=0,
                 root_kernel_size=1,
                 dilation=1,
                 root_residual=False):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(
                cfg, in_channels, out_channels, stride, dilation=dilation)
            self.tree2 = block(
                cfg, out_channels, out_channels, 1, dilation=dilation)
        else:
            self.tree1 = Tree(
                cfg,
                levels - 1,
                block,
                in_channels,
                out_channels,
                stride,
                root_dim=0,
                root_kernel_size=root_kernel_size,
                dilation=dilation,
                root_residual=root_residual)
            self.tree2 = Tree(
                cfg,
                levels - 1,
                block,
                out_channels,
                out_channels,
                root_dim=root_dim + out_channels,
                root_kernel_size=root_kernel_size,
                dilation=dilation,
                root_residual=root_residual)
        if levels == 1:
            self.root = Root(cfg, root_dim, out_channels, root_kernel_size,
                             root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    bias=False), get_norm(cfg.MODEL.DLA.NORM, out_channels))

    def forward(self, x, residual=None, children=None):
        if self.training and residual is not None:
            x = x + residual.sum() * 0.0
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
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


class DLA(Backbone):

    def __init__(self,
                 cfg,
                 levels,
                 channels,
                 block=BasicBlock,
                 residual_root=False):
        super(DLA, self).__init__()
        self.cfg = cfg
        self.channels = channels

        self._out_features = ['dla{}'.format(i) for i in range(6)]
        self._out_feature_channels = {
            k: channels[i]
            for i, k in enumerate(self._out_features)
        }
        self._out_feature_strides = {
            k: 2**i
            for i, k in enumerate(self._out_features)
        }

        self.base_layer = nn.Sequential(
            nn.Conv2d(
                3, channels[0], kernel_size=7, stride=1, padding=3,
                bias=False), get_norm(cfg.MODEL.DLA.NORM, channels[0]),
            nn.ReLU(inplace=True))
        self.level0 = self._make_conv_level(channels[0], channels[0],
                                            levels[0])
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], stride=2)
        self.level2 = Tree(
            cfg,
            levels[2],
            block,
            channels[1],
            channels[2],
            2,
            level_root=False,
            root_residual=residual_root)
        self.level3 = Tree(
            cfg,
            levels[3],
            block,
            channels[2],
            channels[3],
            2,
            level_root=True,
            root_residual=residual_root)
        self.level4 = Tree(
            cfg,
            levels[4],
            block,
            channels[3],
            channels[4],
            2,
            level_root=True,
            root_residual=residual_root)
        self.level5 = Tree(
            cfg,
            levels[5],
            block,
            channels[4],
            channels[5],
            2,
            level_root=True,
            root_residual=residual_root)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

        self.load_pretrained_model(
            data='imagenet', name='dla34', hash='ba72cf86')

    def load_pretrained_model(self, data, name, hash):
        model_url = get_model_url(data, name, hash)
        model_weights = model_zoo.load_url(model_url)
        del model_weights['fc.weight']
        del model_weights['fc.bias']
        print('Loading pretrained DLA!')
        self.load_state_dict(model_weights, strict=True)

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(
                    inplanes,
                    planes,
                    kernel_size=3,
                    stride=stride if i == 0 else 1,
                    padding=dilation,
                    bias=False,
                    dilation=dilation),
                get_norm(self.cfg.MODEL.DLA.NORM, planes),
                nn.ReLU(inplace=True)
            ])
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x):
        y = {}
        x = self.base_layer(x)
        for i in range(6):
            name = 'level{}'.format(i)
            x = getattr(self, name)(x)
            y['dla{}'.format(i)] = x
        return y


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class Conv(nn.Module):

    def __init__(self, chi, cho, norm):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(chi, cho, kernel_size=1, stride=1, bias=False),
            get_norm(norm, cho), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


class DeformConv(nn.Module):

    def __init__(self, chi, cho, norm):
        super(DeformConv, self).__init__()
        self.actf = nn.Sequential(get_norm(norm, cho), nn.ReLU(inplace=True))
        self.offset = Conv2d(
            chi, 27, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv = ModulatedDeformConv(
            chi,
            cho,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            deformable_groups=1)
        nn.init.constant_(self.offset.weight, 0)
        nn.init.constant_(self.offset.bias, 0)

    def forward(self, x):
        offset_mask = self.offset(x)
        offset_x, offset_y, mask = torch.chunk(offset_mask, 3, dim=1)
        offset = torch.cat((offset_x, offset_y), dim=1)
        mask = mask.sigmoid()
        x = self.conv(x, offset, mask)
        x = self.actf(x)
        return x


class IDAUp(nn.Module):

    def __init__(self, o, channels, up_f, norm='FrozenBN', node_type=Conv):
        super(IDAUp, self).__init__()
        for i in range(1, len(channels)):
            c = channels[i]
            f = int(up_f[i])
            proj = node_type(c, o, norm)
            node = node_type(o, o, norm)

            up = nn.ConvTranspose2d(
                o,
                o,
                f * 2,
                stride=f,
                padding=f // 2,
                output_padding=0,
                groups=o,
                bias=False)
            fill_up_weights(up)

            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)

    def forward(self, layers, startp, endp):
        for i in range(startp + 1, endp):
            upsample = getattr(self, 'up_' + str(i - startp))
            project = getattr(self, 'proj_' + str(i - startp))
            layers[i] = upsample(project(layers[i]))
            node = getattr(self, 'node_' + str(i - startp))
            layers[i] = node(layers[i] + layers[i - 1])


DLAUP_NODE_MAP = {
    'conv': Conv,
    'dcn': DeformConv,
}


class DLAUP(Backbone):

    def __init__(self, bottom_up, in_features, norm, dlaup_node='conv'):
        super(DLAUP, self).__init__()
        assert isinstance(bottom_up, Backbone)
        self.bottom_up = bottom_up
        input_shapes = bottom_up.output_shape()
        in_strides = [input_shapes[f].stride for f in in_features]
        in_channels = [input_shapes[f].channels for f in in_features]
        in_levels = [
            int(math.log2(input_shapes[f].stride)) for f in in_features
        ]
        self.in_features = in_features
        out_features = ['dlaup{}'.format(ll) for ll in in_levels]
        self._out_features = out_features
        self._out_feature_channels = {
            'dlaup{}'.format(ll): in_channels[i]
            for i, ll in enumerate(in_levels)
        }
        self._out_feature_strides = {
            'dlaup{}'.format(ll): 2**ll
            for ll in in_levels
        }

        print('self._out_features', self._out_features)
        print('self._out_feature_channels', self._out_feature_channels)
        print('self._out_feature_strides', self._out_feature_strides)
        self._size_divisibility = 32

        node_type = DLAUP_NODE_MAP[dlaup_node]

        self.startp = int(math.log2(in_strides[0]))
        self.channels = in_channels
        channels = list(in_channels)
        scales = np.array([2**i for i in range(len(out_features))], dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(
                self, 'ida_{}'.format(i),
                IDAUp(
                    channels[j],
                    in_channels[j:],
                    scales[j:] // scales[j],
                    norm=norm,
                    node_type=node_type))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    @property
    def size_divisibility(self):
        return self._size_divisibility

    def forward(self, x):
        bottom_up_features = self.bottom_up(x)
        layers = [bottom_up_features[f] for f in self.in_features]
        out = [layers[-1]]  # start with 32
        for i in range(len(layers) - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            ida(layers, len(layers) - i - 2, len(layers))
            out.insert(0, layers[-1])
        ret = {}
        for k, v in zip(self._out_features, out):
            ret[k] = v
        # import pdb; pdb.set_trace()
        return ret


def dla34(cfg, pretrained=None):  # DLA-34
    model = DLA(
        cfg, [1, 1, 1, 2, 2, 1], [16, 32, 64, 128, 256, 512], block=BasicBlock)
    return model


class LastLevelP6P7(nn.Module):
    """This module is used in RetinaNet to generate extra layers, P6 and P7
    from C5 feature."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.num_levels = 2
        self.in_feature = 'dla5'
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        for module in [self.p6, self.p7]:
            weight_init.c2_xavier_fill(module)

    def forward(self, c5):
        p6 = self.p6(c5)
        p7 = self.p7(F.relu(p6))
        return [p6, p7]


@BACKBONE_REGISTRY.register()
def build_dla_fpn3_backbone(cfg, input_shape: ShapeSpec):
    """
    Args: cfg: a detectron2 CfgNode Returns: backbone (Backbone): backbone
    module, must be a subclass of :class:`Backbone`.
    """

    depth_to_creator = {'dla34': dla34}
    bottom_up = depth_to_creator['dla{}'.format(cfg.MODEL.DLA.NUM_LAYERS)](cfg)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS

    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=None,
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )

    return backbone


@BACKBONE_REGISTRY.register()
def build_dla_fpn5_backbone(cfg, input_shape: ShapeSpec):
    """
    Args: cfg: a detectron2 CfgNode Returns: backbone (Backbone): backbone
    module, must be a subclass of :class:`Backbone`.
    """

    depth_to_creator = {'dla34': dla34}
    bottom_up = depth_to_creator['dla{}'.format(cfg.MODEL.DLA.NUM_LAYERS)](cfg)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    in_channels_top = bottom_up.output_shape()['dla5'].channels

    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelP6P7(in_channels_top, out_channels),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )

    return backbone


@BACKBONE_REGISTRY.register()
def build_dlaup_backbone(cfg, input_shape: ShapeSpec):
    """
    Args: cfg: a detectron2 CfgNode Returns: backbone (Backbone): backbone
    module, must be a subclass of :class:`Backbone`.
    """

    depth_to_creator = {'dla34': dla34}
    bottom_up = depth_to_creator['dla{}'.format(cfg.MODEL.DLA.NUM_LAYERS)](cfg)

    backbone = DLAUP(
        bottom_up=bottom_up,
        in_features=cfg.MODEL.DLA.DLAUP_IN_FEATURES,
        norm=cfg.MODEL.DLA.NORM,
        dlaup_node=cfg.MODEL.DLA.DLAUP_NODE,
    )

    return backbone
