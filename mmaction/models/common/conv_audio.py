import torch.nn as nn
from mmcv.cnn import CONV_LAYERS, kaiming_init
from torch.nn.modules.utils import _pair


@CONV_LAYERS.register_module()
class ConvAudio(nn.Module):
    """Conv2d module for AudioResNet backbone.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int | tuple[int]): Same as nn.Conv2d.
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
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True):
        super().__init__()

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        padding = _pair(dilation)

        assert len(kernel_size) == len(stride) == len(padding) == 2

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.output_padding = (0, 0)
        self.transposed = False

        self.conv_1 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=(kernel_size[0], 1),
            stride=(stride[0], 1),
            dilation=(dilation, 1),
            padding=(dilation, 0),
            bias=bias)

        self.conv_2 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, kernel_size[1]),
            stride=(1, stride[1]),
            dilation=(1, dilation),
            padding=(0, dilation),
            bias=bias)

        self.init_weights()

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The output of the module.
        """
        x = self.conv_1(x)
        x = self.conv_2(x)
        return x

    def init_weights(self):
        """Initiate the parameters from scratch."""
        kaiming_init(self.conv_1)
        kaiming_init(self.conv_2)
