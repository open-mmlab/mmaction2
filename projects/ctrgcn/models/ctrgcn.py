import torch
import torch.nn as nn
from mmengine.model import BaseModule, ModuleList

from mmaction.models.utils import Graph, unit_tcn
from mmaction.registry import MODELS
from .ctrgcn_utils import MSTCN, unit_ctrgcn


class CTRGCNBlock(BaseModule):

    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 stride=1,
                 residual=True,
                 kernel_size=5,
                 dilations=[1, 2],
                 tcn_dropout=0):
        super(CTRGCNBlock, self).__init__()
        self.gcn1 = unit_ctrgcn(in_channels, out_channels, A)
        self.tcn1 = MSTCN(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilations=dilations,
            residual=False,
            tcn_dropout=tcn_dropout)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(
                in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        y = self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))
        return y


@MODELS.register_module()
class CTRGCN(BaseModule):

    def __init__(self,
                 graph_cfg,
                 in_channels=3,
                 base_channels=64,
                 num_stages=10,
                 inflate_stages=[5, 8],
                 down_stages=[5, 8],
                 pretrained=None,
                 num_person=2,
                 **kwargs):
        super(CTRGCN, self).__init__()

        self.graph = Graph(**graph_cfg)
        A = torch.tensor(
            self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        self.num_person = num_person
        self.base_channels = base_channels

        self.data_bn = nn.BatchNorm1d(num_person * in_channels * A.size(1))

        kwargs0 = {k: v for k, v in kwargs.items() if k != 'tcn_dropout'}
        modules = [
            CTRGCNBlock(
                in_channels,
                base_channels,
                A.clone(),
                residual=False,
                **kwargs0)
        ]
        for i in range(2, num_stages + 1):
            in_channels = base_channels
            out_channels = base_channels * (1 + (i in inflate_stages))
            stride = 1 + (i in down_stages)
            modules.append(
                CTRGCNBlock(
                    base_channels,
                    out_channels,
                    A.clone(),
                    stride=stride,
                    **kwargs))
            base_channels = out_channels
        self.net = ModuleList(modules)

    def forward(self, x):
        N, M, T, V, C = x.size()
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = self.data_bn(x.view(N, M * V * C, T))
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4,
                                          2).contiguous().view(N * M, C, T, V)

        for gcn in self.net:
            x = gcn(x)

        x = x.reshape((N, M) + x.shape[1:])
        return x
