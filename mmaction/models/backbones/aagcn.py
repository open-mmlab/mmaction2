# # Copyright (c) OpenMMLab. All rights reserved.
# import copy as cp
#
# import torch
# import torch.nn as nn
# from mmcv.runner import load_checkpoint
#
# from ...utils import Graph, cache_checkpoint
# from ..builder import BACKBONES
# from .utils import bn_init, mstcn, unit_aagcn, unit_tcn
#
#
# class AAGCNBlock(nn.Module):
#     """The basic block of AAGCN.
#
#     Args:
#         in_channels (int): Number of input channels.
#         out_channels (int): Number of output channels.
#         A (torch.Tensor): The adjacency matrix defined in the graph
#             with shape of `(num_subsets, num_nodes, num_nodes)`.
#         stride (int): Stride of the temporal convolution. Defaults to 1.
#         residual (bool): Whether to use residual connection. Defaults to True.
#     """
#
#     def __init__(self,
#                  in_channels: int,
#                  out_channels: int,
#                  A: torch.Tensor,
#                  stride: int = 1,
#                  residual: bool = True,
#                  **kwargs) -> None:
#         super().__init__()
#
#         gcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'gcn_'}
#         tcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'tcn_'}
#         kwargs = {k: v for k, v in kwargs.items() if k[:4] not in ['gcn_', 'tcn_']}
#         assert len(kwargs) == 0, f'Invalid arguments: {kwargs}'
#
#         tcn_type = tcn_kwargs.pop('type', 'unit_tcn')
#         assert tcn_type in ['unit_tcn', 'mstcn']
#         gcn_type = gcn_kwargs.pop('type', 'unit_aagcn')
#         assert gcn_type in ['unit_aagcn']
#
#         self.gcn = unit_aagcn(in_channels, out_channels, A, **gcn_kwargs)
#
#         if tcn_type == 'unit_tcn':
#             self.tcn = unit_tcn(out_channels, out_channels, 9, stride=stride, **tcn_kwargs)
#         elif tcn_type == 'mstcn':
#             self.tcn = mstcn(out_channels, out_channels, stride=stride, **tcn_kwargs)
#         self.relu = nn.ReLU()
#
#         if not residual:
#             self.residual = lambda x: 0
#         elif (in_channels == out_channels) and (stride == 1):
#             self.residual = lambda x: x
#         else:
#             self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)
#
#     def init_weights(self):
#         self.tcn.init_weights()
#         self.gcn.init_weights()
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """Defines the computation performed at every call."""
#         return self.relu(self.tcn(self.gcn(x)) + self.residual(x))
#
#
# @BACKBONES.register_module()
# class AAGCN(nn.Module):
#     def __init__(self,
#                  graph_cfg,
#                  in_channels=3,
#                  base_channels=64,
#                  data_bn_type='MVC',
#                  num_person=2,
#                  num_stages=10,
#                  inflate_stages=[5, 8],
#                  down_stages=[5, 8],
#                  pretrained=None,
#                  **kwargs):
#         super().__init__()
#
#         self.graph = Graph(**graph_cfg)
#         A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
#         self.register_buffer('A', A)
#         self.kwargs = kwargs
#
#         assert data_bn_type in ['MVC', 'VC', None]
#         self.data_bn_type = data_bn_type
#         self.in_channels = in_channels
#         self.base_channels = base_channels
#         self.num_person = num_person
#         self.num_stages = num_stages
#         self.inflate_stages = inflate_stages
#         self.down_stages = down_stages
#
#         if self.data_bn_type == 'MVC':
#             self.data_bn = nn.BatchNorm1d(num_person * in_channels * A.size(1))
#         elif self.data_bn_type == 'VC':
#             self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
#         else:
#             self.data_bn = nn.Identity()
#
#         lw_kwargs = [cp.deepcopy(kwargs) for i in range(num_stages)]
#         for k, v in kwargs.items():
#             if isinstance(v, tuple) and len(v) == num_stages:
#                 for i in range(num_stages):
#                     lw_kwargs[i][k] = v[i]
#         lw_kwargs[0].pop('tcn_dropout', None)
#
#         modules = []
#         if self.in_channels != self.base_channels:
#             modules = [AAGCNBlock(in_channels, base_channels, A.clone(), 1, residual=False, **lw_kwargs[0])]
#
#         for i in range(2, num_stages + 1):
#             in_channels = base_channels
#             out_channels = base_channels * (1 + (i in inflate_stages))
#             stride = 1 + (i in down_stages)
#             modules.append(AAGCNBlock(base_channels, out_channels, A.clone(), stride=stride, **lw_kwargs[i - 1]))
#             base_channels = out_channels
#
#         if self.in_channels == self.base_channels:
#             self.num_stages -= 1
#
#         self.gcn = nn.ModuleList(modules)
#         self.pretrained = pretrained
#
#     def init_weights(self):
#         bn_init(self.data_bn, 1)
#         for module in self.gcn:
#             module.init_weights()
#         if isinstance(self.pretrained, str):
#             self.pretrained = cache_checkpoint(self.pretrained)
#             load_checkpoint(self, self.pretrained, strict=False)
#
#     def forward(self, x):
#         N, M, T, V, C = x.size()
#         x = x.permute(0, 1, 3, 4, 2).contiguous()
#         if self.data_bn_type == 'MVC':
#             x = self.data_bn(x.view(N, M * V * C, T))
#         else:
#             x = self.data_bn(x.view(N * M, V * C, T))
#
#         x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
#
#         for i in range(self.num_stages):
#             x = self.gcn[i](x)
#
#         x = x.reshape((N, M) + x.shape[1:])
#         return x
