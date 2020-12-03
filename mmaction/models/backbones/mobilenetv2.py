import torch.nn as nn
from mmcv.cnn import constant_init, kaiming_init


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True))


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True))


def make_divisible(v, divisor, min_value=None):
    """This function is taken from the original tf repo.

    It ensures that all layers have a channel number that is divisible by 8.
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py # noqa
    Args:
        v (float): original number of channels.
        divisor (float): Round the number of channels in each layer to
            be a multiple of this number. Set to 1 to turn off rounding.
        min_value (int): minimal value to return
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride, expand_ratio):
        """Inverted Residual Mobule from MobilNetV2.

        Args:
            inp (int): number of input channels.
            oup (int): number of output channels.
            stride (int): stride for depthwise convolution.
            expand_ratio (int): expand ratio for hidden layers.
        """
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    3,
                    stride,
                    1,
                    groups=hidden_dim,
                    bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    3,
                    stride,
                    1,
                    groups=hidden_dim,
                    bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):

    def __init__(self,
                 width_mult=1.,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 block=None,
                 pretrained=False):
        """MobileNet V2 main class.

        Args:
            width_mult (float): Width multiplier - adjusts number of channels
                in each layer by this amount.
            inverted_residual_setting (list): Network structure.
            round_nearest (int): Round the number of channels in each layer to
                be a multiple of this number. Set to 1 to turn off rounding.
            block (nn.Module): Module specifying inverted residual building
                block for mobilenet.
            pretrained (bool): whether to load pretrained checkpoints.
        """
        super(MobileNetV2, self).__init__()
        self.pretrained = pretrained

        if abs(width_mult - 1.0) > 1e-5 and pretrained:
            raise ValueError('MobileNetV2 only supports one pretrained model '
                             'with `width_mult=1.0`.')

        if block is None:
            block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t/expand_ratio, c/output_channels, n/num_of_blocks, s/stride
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element
        # assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(
                inverted_residual_setting[0]) != 4:
            raise ValueError('inverted_residual_setting should be non-empty '
                             'or a 4-element list, got {}'.format(
                                 inverted_residual_setting))

        input_channel = make_divisible(input_channel * width_mult,
                                       round_nearest)
        self.last_channel = make_divisible(last_channel * max(1.0, width_mult),
                                           round_nearest)

        # first layer
        self.features = [conv_bn(3, input_channel, 2)]

        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = make_divisible(c * width_mult, round_nearest)

            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(
                    block(
                        input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

    def forward(self, x):
        return self.features(x)

    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""

        if self.pretrained:
            from torch.hub import load_state_dict_from_url
            state_dict = load_state_dict_from_url(
                'https://www.dropbox.com/s/47tyzpofuuyyv1b/mobilenetv2_1.0-f2a8633.pth.tar?dl=1',  # noqa
                progress=True)
            del state_dict['classifier.weight']
            del state_dict['classifier.bias']
            self.load_state_dict(state_dict)
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
