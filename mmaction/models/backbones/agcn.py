# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
from mmengine.logging import MMLogger
from mmengine.model.weight_init import constant_init, kaiming_init, normal_init
from mmengine.runner import load_checkpoint

from mmaction.registry import MODELS
from ..utils import Graph


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    normal_init(weight, mean=0, std=math.sqrt(2. / (n * k1 * k2 * branches)))
    constant_init(conv.bias, 0)


def conv_init(conv):
    kaiming_init(conv.weight)
    constant_init(conv.bias, 0)


def bn_init(bn, scale):
    constant_init(bn.weight, scale)
    constant_init(bn.bias, 0)


def zero(x):
    """return zero."""
    return 0


def identity(x):
    """return input itself."""
    return x


class AGCNBlock(nn.Module):
    """Applies spatial graph convolution and  temporal convolution over an
    input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and
            graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        adj_len (int, optional): The length of the adjacency matrix.
            Default: 17
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism.
            Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)`
            format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out},
            V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V,
            V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]
                `,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 adj_len=17,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(
            in_channels, out_channels, kernel_size[1], adj_len=adj_len)
        self.tcn = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, (kernel_size[0], 1),
                      (stride, 1), padding), nn.BatchNorm2d(out_channels))

        # tcn init
        for m in self.tcn.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

        if not residual:
            self.residual = zero

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = identity

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)), nn.BatchNorm2d(out_channels))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, adj_mat):
        """Defines the computation performed at every call."""
        res = self.residual(x)
        x, adj_mat = self.gcn(x, adj_mat)

        x = self.tcn(x) + res

        return self.relu(x), adj_mat


class ConvTemporalGraphical(nn.Module):
    """The basic module for applying a graph convolution.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution.
            Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides
            of the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        adj_len (int, optional): The length of the adjacency matrix.
            Default: 17
        bias (bool, optional): If ``True``, adds a learnable bias to the
            output. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)`
            format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Output graph sequence in :math:`(N, out_channels, T_{out}
            , V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)
            ` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]
                `,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 adj_len=17,
                 bias=True):
        super().__init__()

        self.kernel_size = kernel_size

        self.PA = nn.Parameter(torch.FloatTensor(3, adj_len, adj_len))
        torch.nn.init.constant_(self.PA, 1e-6)

        self.num_subset = 3
        inter_channels = out_channels // 4
        self.inter_c = inter_channels
        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels))
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

    def forward(self, x, adj_mat):
        """Defines the computation performed at every call."""
        assert adj_mat.size(0) == self.kernel_size

        N, C, T, V = x.size()
        A = adj_mat + self.PA

        y = None
        for i in range(self.num_subset):
            A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(
                N, V, self.inter_c * T)
            A2 = self.conv_b[i](x).view(N, self.inter_c * T, V)
            A1 = self.soft(torch.matmul(A1, A2) / A1.size(-1))  # N V V
            A1 = A1 + A[i]
            A2 = x.view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
            y = z + y if y is not None else z
        y = self.bn(y)
        y += self.down(x)

        return self.relu(y), adj_mat


@MODELS.register_module()
class AGCN(nn.Module):
    """Backbone of Two-Stream Adaptive Graph Convolutional Networks for
    Skeleton-Based Action Recognition.

    Args:
        in_channels (int): Number of channels in the input data.
        graph_cfg (dict): The arguments for building the graph.
        data_bn (bool): If 'True', adds data normalization to the inputs.
            Default: True.
        pretrained (str | None): Name of pretrained model.
        **kwargs (optional): Other parameters for graph convolution units.

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self,
                 in_channels,
                 graph_cfg,
                 data_bn=True,
                 pretrained=None,
                 **kwargs):
        super().__init__()

        # load graph
        self.graph = Graph(**graph_cfg)
        A = torch.tensor(
            self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels *
                                      A.size(1)) if data_bn else identity

        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.agcn_networks = nn.ModuleList((
            AGCNBlock(
                in_channels,
                64,
                kernel_size,
                1,
                adj_len=A.size(1),
                residual=False,
                **kwargs0),
            AGCNBlock(64, 64, kernel_size, 1, adj_len=A.size(1), **kwargs),
            AGCNBlock(64, 64, kernel_size, 1, adj_len=A.size(1), **kwargs),
            AGCNBlock(64, 64, kernel_size, 1, adj_len=A.size(1), **kwargs),
            AGCNBlock(64, 128, kernel_size, 2, adj_len=A.size(1), **kwargs),
            AGCNBlock(128, 128, kernel_size, 1, adj_len=A.size(1), **kwargs),
            AGCNBlock(128, 128, kernel_size, 1, adj_len=A.size(1), **kwargs),
            AGCNBlock(128, 256, kernel_size, 2, adj_len=A.size(1), **kwargs),
            AGCNBlock(256, 256, kernel_size, 1, adj_len=A.size(1), **kwargs),
            AGCNBlock(256, 256, kernel_size, 1, adj_len=A.size(1), **kwargs),
        ))

        self.pretrained = pretrained

    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if isinstance(self.pretrained, str):
            logger = MMLogger.get_current_instance()
            logger.info(f'load model from: {self.pretrained}')

            load_checkpoint(self, self.pretrained, strict=False, logger=logger)

        elif self.pretrained is None:
            pass
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Defines the computation performed at every call.
        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The output of the module.
        """
        # data normalization
        x = x.float()
        n, c, t, v, m = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()  # N M V C T
        x = x.view(n * m, v * c, t)
        x = self.data_bn(x)
        x = x.view(n, m, v, c, t)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(n * m, c, t, v)

        for gcn in self.agcn_networks:
            x, _ = gcn(x, self.A)

        return x
