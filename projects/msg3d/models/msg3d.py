import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule, Sequential

from mmaction.models.utils import Graph
from mmaction.registry import MODELS
from .msg3d_utils import MSGCN, MSTCN, MW_MSG3DBlock


@MODELS.register_module()
class MSG3D(BaseModule):

    def __init__(self,
                 graph_cfg,
                 in_channels=3,
                 base_channels=96,
                 num_gcn_scales=13,
                 num_g3d_scales=6,
                 num_person=2,
                 tcn_dropout=0):
        super().__init__()

        self.graph = Graph(**graph_cfg)
        # Note that A is a 2D tensor
        A = torch.tensor(
            self.graph.A[0], dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        self.num_point = A.shape[-1]
        self.in_channels = in_channels
        self.base_channels = base_channels

        self.data_bn = nn.BatchNorm1d(self.num_point * in_channels *
                                      num_person)
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4

        # r=3 STGC blocks
        self.gcn3d1 = MW_MSG3DBlock(3, c1, A, num_g3d_scales, window_stride=1)
        self.sgcn1 = Sequential(
            MSGCN(num_gcn_scales, 3, c1, A), MSTCN(c1, c1), MSTCN(c1, c1))
        self.sgcn1[-1].act = nn.Identity()
        self.tcn1 = MSTCN(c1, c1, tcn_dropout=tcn_dropout)

        self.gcn3d2 = MW_MSG3DBlock(c1, c2, A, num_g3d_scales, window_stride=2)
        self.sgcn2 = Sequential(
            MSGCN(num_gcn_scales, c1, c1, A), MSTCN(c1, c2, stride=2),
            MSTCN(c2, c2))
        self.sgcn2[-1].act = nn.Identity()
        self.tcn2 = MSTCN(c2, c2, tcn_dropout=tcn_dropout)

        self.gcn3d3 = MW_MSG3DBlock(c2, c3, A, num_g3d_scales, window_stride=2)
        self.sgcn3 = Sequential(
            MSGCN(num_gcn_scales, c2, c2, A), MSTCN(c2, c3, stride=2),
            MSTCN(c3, c3))
        self.sgcn3[-1].act = nn.Identity()
        self.tcn3 = MSTCN(c3, c3, tcn_dropout=tcn_dropout)

    def forward(self, x):
        N, M, T, V, C = x.size()
        x = x.permute(0, 1, 3, 4, 2).contiguous().reshape(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.reshape(N * M, V, C, T).permute(0, 2, 3, 1).contiguous()

        # Apply activation to the sum of the pathways
        x = F.relu(self.sgcn1(x) + self.gcn3d1(x), inplace=True)
        x = self.tcn1(x)

        x = F.relu(self.sgcn2(x) + self.gcn3d2(x), inplace=True)
        x = self.tcn2(x)

        x = F.relu(self.sgcn3(x) + self.gcn3d3(x), inplace=True)
        x = self.tcn3(x)

        # N * M, C, T, V
        return x.reshape((N, M) + x.shape[1:])
