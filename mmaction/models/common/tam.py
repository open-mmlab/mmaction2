# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F


class TAM(nn.Module):
    """Temporal Adaptive Module(TAM) for TANet.

    This module is proposed in `TAM: TEMPORAL ADAPTIVE MODULE FOR VIDEO
    RECOGNITION <https://arxiv.org/pdf/2005.06803>`_

    Args:
        in_channels (int): Channel num of input features.
        num_segments (int): Number of frame segments.
        alpha (int): ```alpha``` in the paper and is the ratio of the
            intermediate channel number to the initial channel number in the
            global branch. Default: 2.
        adaptive_kernel_size (int): ```K``` in the paper and is the size of the
            adaptive kernel size in the global branch. Default: 3.
        beta (int): ```beta``` in the paper and is set to control the model
            complexity in the local branch. Default: 4.
        conv1d_kernel_size (int): Size of the convolution kernel of Conv1d in
            the local branch. Default: 3.
        adaptive_convolution_stride (int): The first dimension of strides in
            the adaptive convolution of ```Temporal Adaptive Aggregation```.
            Default: 1.
        adaptive_convolution_padding (int): The first dimension of paddings in
            the adaptive convolution of ```Temporal Adaptive Aggregation```.
            Default: 1.
        init_std (float): Std value for initiation of `nn.Linear`. Default:
            0.001.
    """

    def __init__(self,
                 in_channels,
                 num_segments,
                 alpha=2,
                 adaptive_kernel_size=3,
                 beta=4,
                 conv1d_kernel_size=3,
                 adaptive_convolution_stride=1,
                 adaptive_convolution_padding=1,
                 init_std=0.001):
        super().__init__()

        assert beta > 0 and alpha > 0
        self.in_channels = in_channels
        self.num_segments = num_segments
        self.alpha = alpha
        self.adaptive_kernel_size = adaptive_kernel_size
        self.beta = beta
        self.conv1d_kernel_size = conv1d_kernel_size
        self.adaptive_convolution_stride = adaptive_convolution_stride
        self.adaptive_convolution_padding = adaptive_convolution_padding
        self.init_std = init_std

        self.G = nn.Sequential(
            nn.Linear(num_segments, num_segments * alpha, bias=False),
            nn.BatchNorm1d(num_segments * alpha), nn.ReLU(inplace=True),
            nn.Linear(num_segments * alpha, adaptive_kernel_size, bias=False),
            nn.Softmax(-1))

        self.L = nn.Sequential(
            nn.Conv1d(
                in_channels,
                in_channels // beta,
                conv1d_kernel_size,
                stride=1,
                padding=conv1d_kernel_size // 2,
                bias=False), nn.BatchNorm1d(in_channels // beta),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels // beta, in_channels, 1, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The output of the module.
        """
        # [n, c, h, w]
        n, c, h, w = x.size()
        num_segments = self.num_segments
        num_batches = n // num_segments
        assert c == self.in_channels

        # [num_batches, c, num_segments, h, w]
        x = x.view(num_batches, num_segments, c, h, w)
        x = x.permute(0, 2, 1, 3, 4).contiguous()

        # [num_batches * c, num_segments, 1, 1]
        theta_out = F.adaptive_avg_pool2d(
            x.view(-1, num_segments, h, w), (1, 1))

        # [num_batches * c, 1, adaptive_kernel_size, 1]
        conv_kernel = self.G(theta_out.view(-1, num_segments)).view(
            num_batches * c, 1, -1, 1)

        # [num_batches, c, num_segments, 1, 1]
        local_activation = self.L(theta_out.view(-1, c, num_segments)).view(
            num_batches, c, num_segments, 1, 1)

        # [num_batches, c, num_segments, h, w]
        new_x = x * local_activation

        # [1, num_batches * c, num_segments, h * w]
        y = F.conv2d(
            new_x.view(1, num_batches * c, num_segments, h * w),
            conv_kernel,
            bias=None,
            stride=(self.adaptive_convolution_stride, 1),
            padding=(self.adaptive_convolution_padding, 0),
            groups=num_batches * c)

        # [n, c, h, w]
        y = y.view(num_batches, c, num_segments, h, w)
        y = y.permute(0, 2, 1, 3, 4).contiguous().view(n, c, h, w)

        return y
