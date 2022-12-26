# Copyright (c) OpenMMLab. All rights reserved.
import copy as cp
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from mmengine.model import BaseModule, ModuleList

from mmaction.registry import MODELS
from ..utils import Graph, unit_aagcn, unit_tcn


class AAGCNBlock(BaseModule):
    """The basic block of AAGCN.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        A (torch.Tensor): The adjacency matrix defined in the graph
            with shape of `(num_subsets, num_nodes, num_nodes)`.
        stride (int): Stride of the temporal convolution. Defaults to 1.
        residual (bool): Whether to use residual connection. Defaults to True.
        init_cfg (dict or list[dict], optional): Config to control
            the initialization. Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 A: torch.Tensor,
                 stride: int = 1,
                 residual: bool = True,
                 init_cfg: Optional[Union[Dict, List[Dict]]] = None,
                 **kwargs) -> None:
        super().__init__(init_cfg=init_cfg)

        gcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'gcn_'}
        tcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'tcn_'}
        kwargs = {
            k: v
            for k, v in kwargs.items() if k[:4] not in ['gcn_', 'tcn_']
        }
        assert len(kwargs) == 0, f'Invalid arguments: {kwargs}'

        tcn_type = tcn_kwargs.pop('type', 'unit_tcn')
        assert tcn_type in ['unit_tcn', 'mstcn']
        gcn_type = gcn_kwargs.pop('type', 'unit_aagcn')
        assert gcn_type in ['unit_aagcn']

        self.gcn = unit_aagcn(in_channels, out_channels, A, **gcn_kwargs)

        if tcn_type == 'unit_tcn':
            self.tcn = unit_tcn(
                out_channels, out_channels, 9, stride=stride, **tcn_kwargs)

        self.relu = nn.ReLU()

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(
                in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call."""
        return self.relu(self.tcn(self.gcn(x)) + self.residual(x))


@MODELS.register_module()
class AAGCN(BaseModule):
    """AAGCN backbone, the attention-enhanced version of 2s-AGCN.

    Skeleton-Based Action Recognition with Multi-Stream
    Adaptive Graph Convolutional Networks.
    More details can be found in the `paper
    <https://arxiv.org/abs/1912.06971>`__ .

    Two-Stream Adaptive Graph Convolutional Networks for
    Skeleton-Based Action Recognition.
    More details can be found in the `paper
    <https://arxiv.org/abs/1805.07694>`__ .

    Args:
        graph_cfg (dict): Config for building the graph.
        in_channels (int): Number of input channels. Defaults to 3.
        base_channels (int): Number of base channels. Defaults to 64.
        data_bn_type (str): Type of the data bn layer. Defaults to ``'MVC'``.
        num_person (int): Maximum number of people. Only used when
            data_bn_type == 'MVC'. Defaults to 2.
        num_stages (int): Total number of stages. Defaults to 10.
        inflate_stages (list[int]): Stages to inflate the number of channels.
            Defaults to ``[5, 8]``.
        down_stages (list[int]): Stages to perform downsampling in
            the time dimension. Defaults to ``[5, 8]``.
        init_cfg (dict or list[dict], optional): Config to control
            the initialization. Defaults to None.

        Examples:
        >>> import torch
        >>> from mmaction.models import AAGCN
        >>> from mmaction.utils import register_all_modules
        >>>
        >>> register_all_modules()
        >>> mode = 'stgcn_spatial'
        >>> batch_size, num_person, num_frames = 2, 2, 150
        >>>
        >>> # openpose-18 layout
        >>> num_joints = 18
        >>> model = AAGCN(graph_cfg=dict(layout='openpose', mode=mode))
        >>> model.init_weights()
        >>> inputs = torch.randn(batch_size, num_person,
        ...                      num_frames, num_joints, 3)
        >>> output = model(inputs)
        >>> print(output.shape)
        >>>
        >>> # nturgb+d layout
        >>> num_joints = 25
        >>> model = AAGCN(graph_cfg=dict(layout='nturgb+d', mode=mode))
        >>> model.init_weights()
        >>> inputs = torch.randn(batch_size, num_person,
        ...                      num_frames, num_joints, 3)
        >>> output = model(inputs)
        >>> print(output.shape)
        >>>
        >>> # coco layout
        >>> num_joints = 17
        >>> model = AAGCN(graph_cfg=dict(layout='coco', mode=mode))
        >>> model.init_weights()
        >>> inputs = torch.randn(batch_size, num_person,
        ...                      num_frames, num_joints, 3)
        >>> output = model(inputs)
        >>> print(output.shape)
        >>>
        >>> # custom settings
        >>> # disable the attention module to degenerate AAGCN to AGCN
        >>> model = AAGCN(graph_cfg=dict(layout='coco', mode=mode),
        ...               gcn_attention=False)
        >>> model.init_weights()
        >>> output = model(inputs)
        >>> print(output.shape)
        torch.Size([2, 2, 256, 38, 18])
        torch.Size([2, 2, 256, 38, 25])
        torch.Size([2, 2, 256, 38, 17])
        torch.Size([2, 2, 256, 38, 17])
    """

    def __init__(self,
                 graph_cfg: Dict,
                 in_channels: int = 3,
                 base_channels: int = 64,
                 data_bn_type: str = 'MVC',
                 num_person: int = 2,
                 num_stages: int = 10,
                 inflate_stages: List[int] = [5, 8],
                 down_stages: List[int] = [5, 8],
                 init_cfg: Optional[Union[Dict, List[Dict]]] = None,
                 **kwargs) -> None:
        super().__init__(init_cfg=init_cfg)

        self.graph = Graph(**graph_cfg)
        A = torch.tensor(
            self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        assert data_bn_type in ['MVC', 'VC', None]
        self.data_bn_type = data_bn_type
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_person = num_person
        self.num_stages = num_stages
        self.inflate_stages = inflate_stages
        self.down_stages = down_stages

        if self.data_bn_type == 'MVC':
            self.data_bn = nn.BatchNorm1d(num_person * in_channels * A.size(1))
        elif self.data_bn_type == 'VC':
            self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        else:
            self.data_bn = nn.Identity()

        lw_kwargs = [cp.deepcopy(kwargs) for i in range(num_stages)]
        for k, v in kwargs.items():
            if isinstance(v, tuple) and len(v) == num_stages:
                for i in range(num_stages):
                    lw_kwargs[i][k] = v[i]
        lw_kwargs[0].pop('tcn_dropout', None)

        modules = []
        if self.in_channels != self.base_channels:
            modules = [
                AAGCNBlock(
                    in_channels,
                    base_channels,
                    A.clone(),
                    1,
                    residual=False,
                    **lw_kwargs[0])
            ]

        for i in range(2, num_stages + 1):
            in_channels = base_channels
            out_channels = base_channels * (1 + (i in inflate_stages))
            stride = 1 + (i in down_stages)
            modules.append(
                AAGCNBlock(
                    base_channels,
                    out_channels,
                    A.clone(),
                    stride=stride,
                    **lw_kwargs[i - 1]))
            base_channels = out_channels

        if self.in_channels == self.base_channels:
            self.num_stages -= 1

        self.gcn = ModuleList(modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call."""
        N, M, T, V, C = x.size()
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        if self.data_bn_type == 'MVC':
            x = self.data_bn(x.view(N, M * V * C, T))
        else:
            x = self.data_bn(x.view(N * M, V * C, T))

        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4,
                                          2).contiguous().view(N * M, C, T, V)

        for i in range(self.num_stages):
            x = self.gcn[i](x)

        x = x.reshape((N, M) + x.shape[1:])
        return x
