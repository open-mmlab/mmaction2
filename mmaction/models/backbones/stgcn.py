# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple, Dict, Optional, List, Union
import copy as cp

import torch
import torch.nn as nn
from mmengine.logging import MMLogger
from mmengine.model import BaseModule
from mmengine.model.weight_init import constant_init, kaiming_init, normal_init
from mmengine.runner import load_checkpoint
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm

from mmaction.registry import MODELS
from ..utils import Graph


# class STGCNBlock(nn.Module):
#     """Applies a spatial temporal graph convolution over an input graph
#     sequence.
#
#     Args:
#         in_channels (int): Number of channels in the input sequence data.
#         out_channels (int): Number of channels produced by the convolution.
#         kernel_size (Tuple[int]): Size of the temporal convolving kernel and
#             graph convolving kernel.
#         stride (int, optional): Stride of the temporal convolution.
#             Default: 1.
#         dropout (float, optional): Dropout rate of the final output.
#             Default: 0.
#         residual (bool, optional): If True, applies a residual mechanism.
#             Default: True.
#     """
#
#     def __init__(self,
#                  in_channels: int,
#                  out_channels: int,
#                  kernel_size: Tuple[int],
#                  stride: int = 1,
#                  dropout: float = 0,
#                  residual: bool = True) -> None:
#         super().__init__()
#
#         assert len(kernel_size) == 2
#         assert kernel_size[0] % 2 == 1
#         padding = ((kernel_size[0] - 1) // 2, 0)
#
#         self.gcn = ConvTemporalGraphical(in_channels, out_channels,
#                                          kernel_size[1])
#         self.tcn = nn.Sequential(
#             nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, (kernel_size[0], 1),
#                       (stride, 1), padding), nn.BatchNorm2d(out_channels),
#             nn.Dropout(dropout, inplace=True))
#
#         if not residual:
#             self.residual = lambda x: 0
#         elif (in_channels == out_channels) and (stride == 1):
#             self.residual = lambda x: x
#         else:
#             self.residual = nn.Sequential(
#                 nn.Conv2d(
#                     in_channels,
#                     out_channels,
#                     kernel_size=1,
#                     stride=(stride, 1)), nn.BatchNorm2d(out_channels))
#
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x: torch.Tensor, adj_mat: torch.Tensor) -> tuple:
#         """Defines the computation performed at every call.
#
#         Args:
#             x (torch.Tensor): Input graph sequence in
#                 :math:`(N, in_channels, T_{in}, V)` format.
#             adj_mat (torch.Tensor): Input graph adjacency matrix in
#                 :math:`(K, V, V)` format.
#
#         Returns:
#             tuple: A tuple of output graph sequence and graph adjacency matrix.
#
#                 - x (torch.Tensor): Output graph sequence in
#                     :math:`(N, out_channels, T_{out}, V)` format.
#                 - adj_mat (torch.Tensor): graph adjacency matrix for
#                     output data in :math:`(K, V, V)` format.
#
#         where
#             :math:`N` is the batch size,
#             :math:`K` is the spatial kernel size, as
#                 :math:`K == kernel_size[1]`,
#             :math:`T_{in}/T_{out}` is a length of input/output sequence,
#             :math:`V` is the number of graph nodes.
#         """
#         res = self.residual(x)
#         x, adj_mat = self.gcn(x, adj_mat)
#         x = self.relu(self.tcn(x) + res)
#
#         return x, adj_mat
#
#
# class ConvTemporalGraphical(nn.Module):
#     """The basic module for applying a graph convolution.
#
#     Args:
#         in_channels (int): Number of channels in the input sequence data.
#         out_channels (int): Number of channels produced by the convolution.
#         kernel_size (int): Size of the graph convolution kernel.
#         t_kernel_size (int): Size of the temporal convolution kernel.
#         t_stride (int, optional): Stride of the temporal convolution.
#             Default: 1.
#         t_padding (int, optional): Temporal zero-padding added to both sides
#             of the input. Default: 0.
#         t_dilation (int, optional): Spacing between temporal kernel elements.
#             Default: 1.
#         bias (bool, optional): If True, adds a learnable bias to the
#             output. Default: True.
#     """
#
#     def __init__(self,
#                  in_channels: int,
#                  out_channels: int,
#                  kernel_size: int,
#                  t_kernel_size: int = 1,
#                  t_stride: int = 1,
#                  t_padding: int = 0,
#                  t_dilation: int = 1,
#                  bias: bool = True) -> None:
#         super().__init__()
#
#         self.kernel_size = kernel_size
#         self.conv = nn.Conv2d(
#             in_channels,
#             out_channels * kernel_size,
#             kernel_size=(t_kernel_size, 1),
#             padding=(t_padding, 0),
#             stride=(t_stride, 1),
#             dilation=(t_dilation, 1),
#             bias=bias)
#
#     def forward(self, x: torch.Tensor, adj_mat: torch.Tensor) -> tuple:
#         """Defines the computation performed at every call.
#
#         Args:
#             x (torch.Tensor): Input graph sequence in
#                 :math:`(N, in_channels, T_{in}, V)` format
#             adj_mat (torch.Tensor): Input graph adjacency matrix in
#                 :math:`(K, V, V)` format.
#
#         Returns:
#             tuple: A tuple of output graph sequence and graph adjacency matrix.
#
#                 - x (Tensor): Output graph sequence in
#                     :math:`(N, out_channels, T_{out}, V)` format.
#                 - adj_mat (Tensor): graph adjacency matrix for output data in
#                     :math:`(K, V, V)` format.
#
#         where
#             :math:`N` is a batch size,
#             :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]
#                 `,
#             :math:`T_{in}/T_{out}` is a length of input/output sequence,
#             :math:`V` is the number of graph nodes.
#         """
#         assert adj_mat.size(0) == self.kernel_size
#
#         x = self.conv(x)
#
#         n, kc, t, v = x.size()
#         x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
#         x = torch.einsum('nkctv,kvw->nctw', (x, adj_mat)).contiguous()
#
#         return x, adj_mat
#
#
# @MODELS.register_module()
# class STGCN(nn.Module):
#     """Backbone of spatial temporal graph convolutional networks.
#
#     Args:
#         in_channels (int): Number of channels of the input data.
#         graph_cfg (dict): The arguments for building the graph.
#         edge_importance_weighting (bool): If ``True``, add a learnable
#             importance weighting to the edges of the graph. Defaults to True.
#         data_bn (bool): If ``True``, adds data normalization to the inputs.
#             Defaults to True.
#         pretrained (str, optional): Path of pretrained model.
#         **kwargs: Keyword parameters passed to graph convolution units.
#
#     Shape:
#         - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
#         - Output: :math:`(N, num_class)` where
#             :math:`N` is a batch size,
#             :math:`T_{in}` is a length of input sequence,
#             :math:`V_{in}` is the number of graph nodes,
#             :math:`M_{in}` is the number of instance in a frame.
#     """
#
#     def __init__(self,
#                  in_channels: int,
#                  graph_cfg: dict,
#                  edge_importance_weighting: bool = True,
#                  data_bn: bool = True,
#                  pretrained: str = None,
#                  **kwargs) -> None:
#         super().__init__()
#
#         # load graph
#         self.graph = Graph(**graph_cfg)
#         A = torch.tensor(
#             self.graph.A, dtype=torch.float32, requires_grad=False)
#         self.register_buffer('A', A)
#
#         # build networks
#         spatial_kernel_size = A.size(0)
#         temporal_kernel_size = 9
#         kernel_size = (temporal_kernel_size, spatial_kernel_size)
#         self.data_bn = nn.BatchNorm1d(in_channels *
#                                       A.size(1)) if data_bn else nn.Identity()
#
#         kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
#         self.st_gcn_networks = nn.ModuleList((
#             STGCNBlock(
#                 in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
#             STGCNBlock(64, 64, kernel_size, 1, **kwargs),
#             STGCNBlock(64, 64, kernel_size, 1, **kwargs),
#             STGCNBlock(64, 64, kernel_size, 1, **kwargs),
#             STGCNBlock(64, 128, kernel_size, 2, **kwargs),
#             STGCNBlock(128, 128, kernel_size, 1, **kwargs),
#             STGCNBlock(128, 128, kernel_size, 1, **kwargs),
#             STGCNBlock(128, 256, kernel_size, 2, **kwargs),
#             STGCNBlock(256, 256, kernel_size, 1, **kwargs),
#             STGCNBlock(256, 256, kernel_size, 1, **kwargs),
#         ))
#
#         # initialize parameters for edge importance weighting
#         if edge_importance_weighting:
#             self.edge_importance = nn.ParameterList([
#                 nn.Parameter(torch.ones(self.A.size()))
#                 for i in self.st_gcn_networks
#             ])
#         else:
#             self.edge_importance = [1 for _ in self.st_gcn_networks]
#
#         self.pretrained = pretrained
#
#     def init_weights(self) -> None:
#         """Initiate the parameters either from existing checkpoint or from
#         scratch."""
#         if isinstance(self.pretrained, str):
#             logger = MMLogger.get_current_instance()
#             logger.info(f'load model from: {self.pretrained}')
#
#             load_checkpoint(self, self.pretrained, strict=False, logger=logger)
#
#         elif self.pretrained is None:
#             for m in self.modules():
#                 if isinstance(m, nn.Conv2d):
#                     kaiming_init(m)
#                 elif isinstance(m, nn.Linear):
#                     normal_init(m)
#                 elif isinstance(m, _BatchNorm):
#                     constant_init(m, 1)
#         else:
#             raise TypeError('pretrained must be a str or None')
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """Defines the computation performed at every call.
#
#         Args:
#             x (torch.Tensor): The input data.
#
#         Returns:
#             torch.Tensor: The output of the module.
#         """
#         # data normalization
#         x = x.float()
#         n, c, t, v, m = x.size()  # bs 3 300 25(17) 2
#         x = x.permute(0, 4, 3, 1, 2).contiguous()  # N M V C T
#         x = x.view(n * m, v * c, t)
#         x = self.data_bn(x)
#         x = x.view(n, m, v, c, t)
#         x = x.permute(0, 1, 3, 4, 2).contiguous()
#         x = x.view(n * m, c, t, v)  # bsx2 3 300 25(17)
#
#         # forward
#         for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
#             x, _ = gcn(x, self.A * importance)
#
#         return x


EPS = 1e-4


class STGCNBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 stride=1,
                 residual=True,
                 **kwargs):
        super().__init__()

        gcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'gcn_'}
        tcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'tcn_'}
        kwargs = {k: v for k, v in kwargs.items() if k[:4] not in ['gcn_', 'tcn_']}
        assert len(kwargs) == 0, f'Invalid arguments: {kwargs}'

        tcn_type = tcn_kwargs.pop('type', 'unit_tcn')
        assert tcn_type in ['unit_tcn', 'mstcn']
        gcn_type = gcn_kwargs.pop('type', 'unit_gcn')
        assert gcn_type in ['unit_gcn']

        self.gcn = unit_gcn(in_channels, out_channels, A, **gcn_kwargs)

        if tcn_type == 'unit_tcn':
            self.tcn = unit_tcn(out_channels, out_channels, 9, stride=stride, **tcn_kwargs)
        elif tcn_type == 'mstcn':
            self.tcn = mstcn(out_channels, out_channels, stride=stride, **tcn_kwargs)
        self.relu = nn.ReLU()

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x, A=None):
        """Defines the computation performed at every call."""
        res = self.residual(x)
        x = self.tcn(self.gcn(x, A)) + res
        return self.relu(x)



@MODELS.register_module()
class STGCN(BaseModule):
    """Backbone of spatial temporal graph convolutional networks.

    Args:
        graph_cfg (dict): Config for building the graph.
        in_channels (int): Number of input channels. Defaults to 3.
        base_channels (int): Number of base channels. Defaults to 64.
        data_bn_type (str): Type of the data bn layer. Defaults to ``'VC'``.
        ch_ratio (int): Inflation ratio of the number of channels.
            Defaults to 2.
        num_person (int): Maximum number of people. Only used when
            data_bn_type == 'MVC'. Defaults to 2.
        num_stages (int): Total number of stages. Defaults to 10.
        inflate_stages (list[int]): Stages to inflate the number of channels.
            Defaults to ``[5, 8]``.
        down_stages (list[int]): Stages to perform downsampling in
            the time dimension. Defaults to ``[5, 8]``.
        stage_cfgs (dict): Extra config dict for each stage.
            Defaults to ``dict()``.
        init_cfg (dict or list[dict], optional): Config to control
            the initialization. Defaults to None.
    """

    def __init__(self,
                 graph_cfg: Dict,
                 in_channels: int = 3,
                 base_channels: int = 64,
                 data_bn_type: str = 'VC',
                 ch_ratio: int = 2,
                 num_person: int = 2,
                 num_stages: int = 10,
                 inflate_stages: List[int] = [5, 8],
                 down_stages: List[int] = [5, 8],
                 stage_cfgs: dict = dict(),
                 init_cfg: Optional[Union[Dict, List[Dict]]] = None) -> None:
        super().__init__(init_cfg=init_cfg)

        self.graph = Graph(**graph_cfg)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.data_bn_type = data_bn_type

        if data_bn_type == 'MVC':
            self.data_bn = nn.BatchNorm1d(num_person * in_channels * A.size(1))
        elif data_bn_type == 'VC':
            self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        else:
            self.data_bn = nn.Identity()

        lw_kwargs = [cp.deepcopy(stage_cfgs) for i in range(num_stages)]
        for k, v in stage_cfgs.items():
            if isinstance(v, tuple) and len(v) == num_stages:
                for i in range(num_stages):
                    lw_kwargs[i][k] = v[i]
        lw_kwargs[0].pop('tcn_dropout', None)

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.ch_ratio = ch_ratio
        self.inflate_stages = inflate_stages
        self.down_stages = down_stages

        modules = []
        if self.in_channels != self.base_channels:
            modules = [STGCNBlock(in_channels, base_channels, A.clone(), 1, residual=False, **lw_kwargs[0])]

        inflate_times = 0
        for i in range(2, num_stages + 1):
            stride = 1 + (i in down_stages)
            in_channels = base_channels
            if i in inflate_stages:
                inflate_times += 1
            out_channels = int(self.base_channels * self.ch_ratio ** inflate_times + EPS)
            base_channels = out_channels
            modules.append(STGCNBlock(in_channels, out_channels, A.clone(), stride, **lw_kwargs[i - 1]))

        if self.in_channels == self.base_channels:
            num_stages -= 1

        self.num_stages = num_stages
        self.gcn = nn.ModuleList(modules)

    def forward(self, x):
        N, M, T, V, C = x.size()
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        if self.data_bn_type == 'MVC':
            x = self.data_bn(x.view(N, M * V * C, T))
        else:
            x = self.data_bn(x.view(N * M, V * C, T))
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        for i in range(self.num_stages):
            x = self.gcn[i](x)

        x = x.reshape((N, M) + x.shape[1:])
        return x
