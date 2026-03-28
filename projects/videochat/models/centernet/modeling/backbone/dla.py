import math
from os.path import join

import fvcore.nn.weight_init as weight_init
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from detectron2.layers import (Conv2d, DeformConv, ModulatedDeformConv,
                               ShapeSpec, get_norm)
from detectron2.modeling.backbone.backbone import Backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone.fpn import FPN
from detectron2.modeling.backbone.resnet import (BasicStem, BottleneckBlock,
                                                 DeformBottleneckBlock)
from torch import nn

__all__ = [
    'BottleneckBlock',
    'DeformBottleneckBlock',
    'BasicStem',
]

DCNV1 = False

HASH = {
    34: 'ba72cf86',
    60: '24839fc4',
}


def get_model_url(data, name, hash):
    return join('http://dl.yf.io/dla/models', data,
                '{}-{}.pth'.format(name, hash))


class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, dilation=1, norm='BN'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            bias=False,
            dilation=dilation)
        self.bn1 = get_norm(norm, planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=dilation,
            bias=False,
            dilation=dilation)
        self.bn2 = get_norm(norm, planes)
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

    def __init__(self, inplanes, planes, stride=1, dilation=1, norm='BN'):
        super(Bottleneck, self).__init__()
        expansion = Bottleneck.expansion
        bottle_planes = planes // expansion
        self.conv1 = nn.Conv2d(
            inplanes, bottle_planes, kernel_size=1, bias=False)
        self.bn1 = get_norm(norm, bottle_planes)
        self.conv2 = nn.Conv2d(
            bottle_planes,
            bottle_planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            bias=False,
            dilation=dilation)
        self.bn2 = get_norm(norm, bottle_planes)
        self.conv3 = nn.Conv2d(
            bottle_planes, planes, kernel_size=1, bias=False)
        self.bn3 = get_norm(norm, planes)
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

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 residual,
                 norm='BN'):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            1,
            stride=1,
            bias=False,
            padding=(kernel_size - 1) // 2)
        self.bn = get_norm(norm, out_channels)
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
                 levels,
                 block,
                 in_channels,
                 out_channels,
                 stride=1,
                 level_root=False,
                 root_dim=0,
                 root_kernel_size=1,
                 dilation=1,
                 root_residual=False,
                 norm='BN'):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(
                in_channels,
                out_channels,
                stride,
                dilation=dilation,
                norm=norm)
            self.tree2 = block(
                out_channels, out_channels, 1, dilation=dilation, norm=norm)
        else:
            self.tree1 = Tree(
                levels - 1,
                block,
                in_channels,
                out_channels,
                stride,
                root_dim=0,
                root_kernel_size=root_kernel_size,
                dilation=dilation,
                root_residual=root_residual,
                norm=norm)
            self.tree2 = Tree(
                levels - 1,
                block,
                out_channels,
                out_channels,
                root_dim=root_dim + out_channels,
                root_kernel_size=root_kernel_size,
                dilation=dilation,
                root_residual=root_residual,
                norm=norm)
        if levels == 1:
            self.root = Root(
                root_dim,
                out_channels,
                root_kernel_size,
                root_residual,
                norm=norm)
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
                    bias=False), get_norm(norm, out_channels))

    def forward(self, x, residual=None, children=None):
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


class DLA(nn.Module):

    def __init__(self,
                 num_layers,
                 levels,
                 channels,
                 block=BasicBlock,
                 residual_root=False,
                 norm='BN'):
        """
        Args:
        """
        super(DLA, self).__init__()
        self.norm = norm
        self.channels = channels
        self.base_layer = nn.Sequential(
            nn.Conv2d(
                3, channels[0], kernel_size=7, stride=1, padding=3,
                bias=False), get_norm(self.norm, channels[0]),
            nn.ReLU(inplace=True))
        self.level0 = self._make_conv_level(channels[0], channels[0],
                                            levels[0])
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], stride=2)
        self.level2 = Tree(
            levels[2],
            block,
            channels[1],
            channels[2],
            2,
            level_root=False,
            root_residual=residual_root,
            norm=norm)
        self.level3 = Tree(
            levels[3],
            block,
            channels[2],
            channels[3],
            2,
            level_root=True,
            root_residual=residual_root,
            norm=norm)
        self.level4 = Tree(
            levels[4],
            block,
            channels[3],
            channels[4],
            2,
            level_root=True,
            root_residual=residual_root,
            norm=norm)
        self.level5 = Tree(
            levels[5],
            block,
            channels[4],
            channels[5],
            2,
            level_root=True,
            root_residual=residual_root,
            norm=norm)
        self.load_pretrained_model(
            data='imagenet',
            name='dla{}'.format(num_layers),
            hash=HASH[num_layers])

    def load_pretrained_model(self, data, name, hash):
        model_url = get_model_url(data, name, hash)
        model_weights = model_zoo.load_url(model_url)
        num_classes = len(model_weights[list(model_weights.keys())[-1]])
        self.fc = nn.Conv2d(
            self.channels[-1],
            num_classes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)
        print('Loading pretrained')
        self.load_state_dict(model_weights, strict=False)

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
                get_norm(self.norm, planes),
                nn.ReLU(inplace=True)
            ])
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x):
        y = []
        x = self.base_layer(x)
        for i in range(6):
            x = getattr(self, 'level{}'.format(i))(x)
            y.append(x)
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


class _DeformConv(nn.Module):

    def __init__(self, chi, cho, norm='BN'):
        super(_DeformConv, self).__init__()
        self.actf = nn.Sequential(get_norm(norm, cho), nn.ReLU(inplace=True))
        if DCNV1:
            self.offset = Conv2d(
                chi, 18, kernel_size=3, stride=1, padding=1, dilation=1)
            self.conv = DeformConv(
                chi,
                cho,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                dilation=1,
                deformable_groups=1)
        else:
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
        if DCNV1:
            offset = self.offset(x)
            x = self.conv(x, offset)
        else:
            offset_mask = self.offset(x)
            offset_x, offset_y, mask = torch.chunk(offset_mask, 3, dim=1)
            offset = torch.cat((offset_x, offset_y), dim=1)
            mask = mask.sigmoid()
            x = self.conv(x, offset, mask)
        x = self.actf(x)
        return x


class IDAUp(nn.Module):

    def __init__(self, o, channels, up_f, norm='BN'):
        super(IDAUp, self).__init__()
        for i in range(1, len(channels)):
            c = channels[i]
            f = int(up_f[i])
            proj = _DeformConv(c, o, norm=norm)
            node = _DeformConv(o, o, norm=norm)

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


class DLAUp(nn.Module):

    def __init__(self, startp, channels, scales, in_channels=None, norm='BN'):
        super(DLAUp, self).__init__()
        self.startp = startp
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(
                self, 'ida_{}'.format(i),
                IDAUp(
                    channels[j],
                    in_channels[j:],
                    scales[j:] // scales[j],
                    norm=norm))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, layers):
        out = [layers[-1]]  # start with 32
        for i in range(len(layers) - self.startp - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            ida(layers, len(layers) - i - 2, len(layers))
            out.insert(0, layers[-1])
        return out


DLA_CONFIGS = {
    34: ([1, 1, 1, 2, 2, 1], [16, 32, 64, 128, 256, 512], BasicBlock),
    60: ([1, 1, 1, 2, 3, 1], [16, 32, 128, 256, 512, 1024], Bottleneck)
}


class DLASeg(Backbone):

    def __init__(self,
                 num_layers,
                 out_features,
                 use_dla_up=True,
                 ms_output=False,
                 norm='BN'):
        super(DLASeg, self).__init__()
        # depth = 34
        levels, channels, Block = DLA_CONFIGS[num_layers]
        self.base = DLA(
            num_layers=num_layers,
            levels=levels,
            channels=channels,
            block=Block,
            norm=norm)
        down_ratio = 4
        self.first_level = int(np.log2(down_ratio))
        self.ms_output = ms_output
        self.last_level = 5 if not self.ms_output else 6
        channels = self.base.channels
        scales = [2**i for i in range(len(channels[self.first_level:]))]
        self.use_dla_up = use_dla_up
        if self.use_dla_up:
            self.dla_up = DLAUp(
                self.first_level,
                channels[self.first_level:],
                scales,
                norm=norm)
        out_channel = channels[self.first_level]
        if not self.ms_output:  # stride 4 DLA
            self.ida_up = IDAUp(
                out_channel,
                channels[self.first_level:self.last_level],
                [2**i for i in range(self.last_level - self.first_level)],
                norm=norm)
        self._out_features = out_features
        self._out_feature_channels = {
            'dla{}'.format(i): channels[i]
            for i in range(6)
        }
        self._out_feature_strides = {'dla{}'.format(i): 2**i for i in range(6)}
        self._size_divisibility = 32

    @property
    def size_divisibility(self):
        return self._size_divisibility

    def forward(self, x):
        x = self.base(x)
        if self.use_dla_up:
            x = self.dla_up(x)
        if not self.ms_output:  # stride 4 dla
            y = []
            for i in range(self.last_level - self.first_level):
                y.append(x[i].clone())
            self.ida_up(y, 0, len(y))
            ret = {}
            for i in range(self.last_level - self.first_level):
                out_feature = 'dla{}'.format(i)
                if out_feature in self._out_features:
                    ret[out_feature] = y[i]
        else:
            ret = {}
            st = self.first_level if self.use_dla_up else 0
            for i in range(self.last_level - st):
                out_feature = 'dla{}'.format(i + st)
                if out_feature in self._out_features:
                    ret[out_feature] = x[i]

        return ret


@BACKBONE_REGISTRY.register()
def build_dla_backbone(cfg, input_shape):
    """Create a ResNet instance from config.

    Returns:
        ResNet: a :class:`ResNet` instance.
    """
    return DLASeg(
        out_features=cfg.MODEL.DLA.OUT_FEATURES,
        num_layers=cfg.MODEL.DLA.NUM_LAYERS,
        use_dla_up=cfg.MODEL.DLA.USE_DLA_UP,
        ms_output=cfg.MODEL.DLA.MS_OUTPUT,
        norm=cfg.MODEL.DLA.NORM)


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
def build_retinanet_dla_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args: cfg: a detectron2 CfgNode Returns: backbone (Backbone): backbone
    module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_dla_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    in_channels_p6p7 = bottom_up.output_shape()['dla5'].channels
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelP6P7(in_channels_p6p7, out_channels),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone
