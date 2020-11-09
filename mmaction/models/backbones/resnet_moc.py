import math

from mmcv.ops import DeformConv2d
from torch import nn as nn

from ..registry import BACKBONES
from .resnet import ResNet


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


@BACKBONES.register_module()
class ResNetMOC(ResNet):

    deconv_cfg = {4: (4, 1, 0), 3: (3, 1, 1), 2: (2, 0, 0)}

    def __init__(self,
                 *args,
                 deconv_channels=(256, 128, 64),
                 deconv_kernel_size=(4, 4, 4),
                 **kwargs):
        super().__init__(*args, *kwargs)

        if not isinstance(deconv_channels, (tuple, list)):
            raise TypeError('deconv channels should be tuple or list, '
                            f'but got {type(deconv_channels)}')

        if isinstance(deconv_kernel_size, int):
            deconv_kernel_size = [deconv_kernel_size for _ in deconv_channels]

        assert len(deconv_channels) == len(deconv_kernel_size)

        self.deconv_channels = deconv_channels
        self.deconv_kernel_size = deconv_kernel_size

    def _make_deconv_layer(self):
        layers = []
        for channel, kernel_size in zip(self.deconv_channels,
                                        self.deconv_kernel_size):
            kernel, padding, output_padding = self.deconv_cfg[kernel_size]
            fc = DeformConv2d(
                self.inplanes,
                channel,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
                deform_groups=1)
            up = nn.ConvTranspose2d(
                channel,
                channel,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
                output_padding=output_padding,
                bias=False)
            fill_up_weights(up)

            layers.extend([fc, nn.BatchNorm2d(channel), nn.ReLU(inplace=True)])
            layers.extend([up, nn.BatchNorm2d(channel), nn.ReLU(inplace=True)])

            self.inplanes = channel

        return nn.Sequential(*layers)
