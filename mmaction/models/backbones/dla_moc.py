import math

import numpy as np
from mmcv.ops import DeformConv2d
from torch import nn as nn

from ..registry import BACKBONES
from .dla import DLA


def fill_up_weights(up_module):
    w = up_module.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i,
              j] = (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, ...] = w[0, 0, ...]


class DeformConv(nn.Module):

    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.act = nn.Sequential(
            nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True))
        self.conv = DeformConv2d(
            in_channel,
            out_channel,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            dilation=1,
            deform_groups=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        return x


class IDAUp(nn.Module):

    def __init__(self, out_channel, channels, up_filters):
        super().__init__()
        self.proj_layers = []
        self.up_layers = []
        self.node_layers = []

        for i in range(1, len(channels)):
            channel = channels[i]
            up_filter = up_filters[i]
            project = DeformConv(channel, out_channel)
            node = DeformConv(out_channel, out_channel)

            up = nn.ConvTranspose2d(
                out_channel,
                out_channel,
                kernel_size=up_filter * 2,
                stride=up_filter,
                padding=up_filter // 2,
                output_padding=0,
                groups=out_channel,
                bias=False)
            fill_up_weights(up)

            project_name = f'proj_{i}'
            up_name = f'up_{i}'
            node_name = f'node_{i}'

            self.add_module(project_name, project)
            self.add_module(up_name, up)
            self.add_module(node_name, node)
            self.proj_layers.append(project_name)
            self.up_layers.append(up_name)
            self.node_layers.append(node_name)

    def forward(self, layers, start_p, end_p):
        for i in range(start_p + 1, end_p):
            upsample = getattr(self, self.up_layers[i - start_p])
            project = getattr(self, self.proj_layers[i - start_p])
            node = getattr(self, self.node_layers[i - start_p])
            layers[i] = upsample(project(layers[i]))
            layers[i] = node(layers[i] + layers[i - 1])


class DLAUp(nn.Module):

    def __init__(self, start_p, channels, scales, in_channels=None):
        super().__init__()
        self.start_p = start_p
        in_channels = channels if in_channels is None else in_channels
        self.channels = channels
        self.layers = []
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            layer_name = f'ida_{i}'
            layer = IDAUp(channels[j], in_channels[j:],
                          scales[j:] // scales[j])
            self.add_module(layer_name, layer)
            self.layers.append(layer_name)

            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, layers):
        out = [layers[-1]]
        for i in range(len(layers) - self.start_p - 1):
            layer_name = self.layers[i]
            layer = getattr(self, layer_name)
            layer(layers, len(layers) - i - 2, len(layers))
            out.insert(0, layers[-1])
        return out


@BACKBONES.register_module()
class DLAMOC(DLA):

    def __init__(self, levels, channels, **kwargs):
        self.output_channel = 64
        self.first_level = 2
        self.last_level = 5

        super().__init__(levels, channels, **kwargs)

        scales = [2**i for i in range(len(self.channels[self.first_level:]))]
        self.dla_up = DLAUp(self.first_level, channels[self.first_level:],
                            scales)

        out_channels = channels[self.first_level]
        self.ida_up = IDAUp(
            out_channels, channels[self.first_level:self.last_level],
            [2**i for i in range(self.last_level - self.first_level)])

    def forward(self, x):
        x = self.base_layer(x)
        x_list = []
        for i in range(6):
            x = getattr(self, f'level{i}')(x)
            x_list.append(x)
        x = self.dla_up(x_list)
        x_list = []
        for i in range(self.last_level - self.first_level):
            x_list.append(x[i].clone())
        self.ida_up(x_list, 0, len(x_list))

        return x_list[-1]
