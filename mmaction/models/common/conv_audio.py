import torch
import torch.nn as nn
from mmcv.cnn import CONV_LAYERS, ConvModule, constant_init, kaiming_init
from torch.nn.modules.utils import _pair


@CONV_LAYERS.register_module()
class ConvAudio(nn.Module):
    """Conv2d module for AudioResNet backbone.

        <https://arxiv.org/abs/2001.08740>`_.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int | tuple[int]): Same as nn.Conv2d.
        op (string): Operation to merge the output of freq
            and time feature map. Choices are 'sum' and 'concat'.
            Default: 'concat'.
        stride (int | tuple[int]): Same as nn.Conv2d.
        padding (int | tuple[int]): Same as nn.Conv2d.
        dilation (int | tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 op='concat',
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=False):
        super().__init__()

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        assert op in ['concat', 'sum']
        self.op = op
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.output_padding = (0, 0)
        self.transposed = False

        self.conv_1 = ConvModule(
            in_channels,
            out_channels,
            kernel_size=(kernel_size[0], 1),
            stride=stride,
            padding=(kernel_size[0] // 2, 0),
            bias=bias,
            conv_cfg=dict(type='Conv'),
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'))

        self.conv_2 = ConvModule(
            in_channels,
            out_channels,
            kernel_size=(1, kernel_size[1]),
            stride=stride,
            padding=(0, kernel_size[1] // 2),
            bias=bias,
            conv_cfg=dict(type='Conv'),
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'))

        self.init_weights()

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The output of the module.
        """
        x_1 = self.conv_1(x)
        x_2 = self.conv_2(x)
        if self.op == 'concat':
            out = torch.cat([x_1, x_2], 1)
        else:
            out = x_1 + x_2
        return out

    def init_weights(self):
        """Initiate the parameters from scratch."""
        kaiming_init(self.conv_1.conv)
        kaiming_init(self.conv_2.conv)
        constant_init(self.conv_1.bn, 1, bias=0)
        constant_init(self.conv_2.bn, 1, bias=0)
