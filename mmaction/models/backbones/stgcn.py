# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch
import torch.nn as nn
from mmengine.logging import MMLogger
from mmengine.model.weight_init import constant_init, kaiming_init, normal_init
from mmengine.runner import load_checkpoint
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm
from torch import Tensor

from mmaction.registry import MODELS
from mmaction.utils import ConfigType
from ..utils import Graph


def zero(x: Tensor) -> int:
    """return zero."""
    return 0


def identity(x: Tensor) -> Tensor:
    """return input itself."""
    return x


class STGCNBlock(nn.Module):
    """Applies a spatial temporal graph convolution over an input graph
    sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (Tuple[int]): Size of the temporal convolving kernel and
            graph convolving kernel.
        stride (int, optional): Stride of the temporal convolution.
            Default: 1.
        dropout (float, optional): Dropout rate of the final output.
            Default: 0.
        residual (bool, optional): If True, applies a residual mechanism.
            Default: True.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int],
                 stride: int = 1,
                 dropout: float = 0,
                 residual: bool = True) -> None:
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, (kernel_size[0], 1),
                      (stride, 1), padding), nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True))

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

    def forward(self, x: Tensor, adj_mat: Tensor) -> tuple:
        """Defines the computation performed at every call.

        Args:
            x (Tensor): Input graph sequence in
                :math:`(N, in_channels, T_{in}, V)` format.
            adj_mat (Tensor): Input graph adjacency matrix in :math:`(K, V, V)`
                format.

        Returns:
            tuple: A tuple of output graph sequence and graph adjacency matrix.

                - x (Tensor): Output graph sequence in
                    :math:`(N, out_channels, T_{out}, V)` format.
                - adj_mat (Tensor): graph adjacency matrix for output data in
                    :math:`(K, V, V)` format.

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]
                `,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
        """
        res = self.residual(x)
        x, adj_mat = self.gcn(x, adj_mat)
        x = self.relu(self.tcn(x) + res)

        return x, adj_mat


class ConvTemporalGraphical(nn.Module):
    """The basic module for applying a graph convolution.

    Args:
        in_channels (int): Number of channels in the input sequence data.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the graph convolving kernel.
        t_kernel_size (int): Size of the temporal convolving kernel.
        t_stride (int, optional): Stride of the temporal convolution.
            Default: 1.
        t_padding (int, optional): Temporal zero-padding added to both sides
            of the input. Default: 0.
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1.
        bias (bool, optional): If True, adds a learnable bias to the
            output. Default: True.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 t_kernel_size: int = 1,
                 t_stride: int = 1,
                 t_padding: int = 0,
                 t_dilation: int = 1,
                 bias: bool = True) -> None:
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

    def forward(self, x: Tensor, adj_mat: Tensor) -> tuple:
        """Defines the computation performed at every call.

        Args:
            x (Tensor): Input graph sequence in
                :math:`(N, in_channels, T_{in}, V)` format
            adj_mat (Tensor): Input graph adjacency matrix in
                :math:`(K, V, V)` format.

        Returns:
            tuple: A tuple of output graph sequence and graph adjacency matrix.

                - x (Tensor): Output graph sequence in
                    :math:`(N, out_channels, T_{out}, V)` format.
                - adj_mat (Tensor): graph adjacency matrix for output data in
                    :math:`(K, V, V)` format.

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]
                `,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
        """
        assert adj_mat.size(0) == self.kernel_size

        x = self.conv(x)

        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, adj_mat))
        x = x.contiguous()

        return x, adj_mat


@MODELS.register_module()
class STGCN(nn.Module):
    """Backbone of Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data.
        graph_cfg (dict or ConfigDict): The arguments for building the graph.
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph. Default: True.
        data_bn (bool): If 'True', adds data normalization to the inputs.
            Default: True.
        pretrained (str or None): Name of pretrained model.

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self,
                 in_channels: int,
                 graph_cfg: ConfigType,
                 edge_importance_weighting: bool = True,
                 data_bn: bool = True,
                 pretrained: str = None,
                 **kwargs) -> None:
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
        self.st_gcn_networks = nn.ModuleList((
            STGCNBlock(
                in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
            STGCNBlock(64, 64, kernel_size, 1, **kwargs),
            STGCNBlock(64, 64, kernel_size, 1, **kwargs),
            STGCNBlock(64, 64, kernel_size, 1, **kwargs),
            STGCNBlock(64, 128, kernel_size, 2, **kwargs),
            STGCNBlock(128, 128, kernel_size, 1, **kwargs),
            STGCNBlock(128, 128, kernel_size, 1, **kwargs),
            STGCNBlock(128, 256, kernel_size, 2, **kwargs),
            STGCNBlock(256, 256, kernel_size, 1, **kwargs),
            STGCNBlock(256, 256, kernel_size, 1, **kwargs),
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1 for _ in self.st_gcn_networks]

        self.pretrained = pretrained

    def init_weights(self) -> None:
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if isinstance(self.pretrained, str):
            logger = MMLogger.get_current_instance()
            logger.info(f'load model from: {self.pretrained}')

            load_checkpoint(self, self.pretrained, strict=False, logger=logger)

        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.Linear):
                    normal_init(m)
                elif isinstance(m, _BatchNorm):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x: Tensor) -> Tensor:
        """Defines the computation performed at every call.
        Args:
            x (Tensor): The input data.

        Returns:
            Tensor: The output of the module.
        """
        # data normalization
        x = x.float()
        n, c, t, v, m = x.size()  # bs 3 300 25(17) 2
        x = x.permute(0, 4, 3, 1, 2).contiguous()  # N M V C T
        x = x.view(n * m, v * c, t)
        x = self.data_bn(x)
        x = x.view(n, m, v, c, t)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(n * m, c, t, v)  # bsx2 3 300 25(17)

        # forward
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        return x
