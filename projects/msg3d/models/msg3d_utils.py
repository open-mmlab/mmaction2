import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import build_activation_layer
from mmengine.model import BaseModule, ModuleList, Sequential

from mmaction.models.utils import unit_tcn
from mmaction.models.utils.graph import k_adjacency, normalize_digraph


class MLP(BaseModule):

    def __init__(self,
                 in_channels,
                 out_channels,
                 act_cfg=dict(type='ReLU'),
                 dropout=0):
        super().__init__()
        channels = [in_channels] + out_channels
        self.layers = ModuleList()
        for i in range(1, len(channels)):
            if dropout > 1e-3:
                self.layers.append(nn.Dropout(p=dropout))
            self.layers.append(
                nn.Conv2d(channels[i - 1], channels[i], kernel_size=1))
            self.layers.append(nn.BatchNorm2d(channels[i]))
            if act_cfg:
                self.layers.append(build_activation_layer(act_cfg))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MSGCN(BaseModule):

    def __init__(self,
                 num_scales,
                 in_channels,
                 out_channels,
                 A,
                 dropout=0,
                 act_cfg=dict(type='ReLU')):
        super().__init__()
        self.num_scales = num_scales

        A_powers = [
            k_adjacency(A, k, with_self=True) for k in range(num_scales)
        ]
        A_powers = np.stack([normalize_digraph(g) for g in A_powers])

        # K, V, V
        self.register_buffer('A', torch.Tensor(A_powers))
        self.PA = nn.Parameter(self.A.clone())
        nn.init.uniform_(self.PA, -1e-6, 1e-6)

        self.mlp = MLP(
            in_channels * num_scales, [out_channels],
            dropout=dropout,
            act_cfg=act_cfg)

    def forward(self, x):
        N, C, T, V = x.shape
        A = self.A
        A = A + self.PA

        support = torch.einsum('kvu,nctv->nkctu', A, x)
        support = support.reshape(N, self.num_scales * C, T, V)
        out = self.mlp(support)
        return out


# ! Notice: The implementation of MSTCN in
# MS-G3D is not the same as our implementation.
class MSTCN(BaseModule):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilations=[1, 2, 3, 4],
                 residual=True,
                 act_cfg=dict(type='ReLU'),
                 init_cfg=[
                     dict(type='Constant', layer='BatchNorm2d', val=1),
                     dict(type='Kaiming', layer='Conv2d', mode='fan_out')
                 ],
                 tcn_dropout=0):

        super().__init__(init_cfg=init_cfg)
        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        branch_channels_rem = out_channels - branch_channels * (
            self.num_branches - 1)

        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size] * len(dilations)

        self.branches = ModuleList([
            Sequential(
                nn.Conv2d(
                    in_channels, branch_channels, kernel_size=1, padding=0),
                nn.BatchNorm2d(branch_channels),
                build_activation_layer(act_cfg),
                unit_tcn(
                    branch_channels,
                    branch_channels,
                    kernel_size=ks,
                    stride=stride,
                    dilation=dilation),
            ) for ks, dilation in zip(kernel_size, dilations)
        ])

        # Additional Max & 1x1 branch
        self.branches.append(
            Sequential(
                nn.Conv2d(
                    in_channels, branch_channels, kernel_size=1, padding=0),
                nn.BatchNorm2d(branch_channels),
                build_activation_layer(act_cfg),
                nn.MaxPool2d(
                    kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
                nn.BatchNorm2d(branch_channels)))

        self.branches.append(
            Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels_rem,
                    kernel_size=1,
                    padding=0,
                    stride=(stride, 1)), nn.BatchNorm2d(branch_channels_rem)))

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(
                in_channels, out_channels, kernel_size=1, stride=stride)

        self.act = build_activation_layer(act_cfg)
        self.drop = nn.Dropout(tcn_dropout)

    def forward(self, x):
        # Input dim: (N,C,T,V)
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        out += res
        out = self.act(out)
        out = self.drop(out)
        return out


class UnfoldTemporalWindows(BaseModule):

    def __init__(self, window_size, window_stride, window_dilation=1):
        super().__init__()
        self.window_size = window_size
        self.window_stride = window_stride
        self.window_dilation = window_dilation

        self.padding = (window_size + (window_size - 1) *
                        (window_dilation - 1) - 1) // 2
        self.unfold = nn.Unfold(
            kernel_size=(self.window_size, 1),
            dilation=(self.window_dilation, 1),
            stride=(self.window_stride, 1),
            padding=(self.padding, 0))

    def forward(self, x):
        # Input shape: (N,C,T,V), out: (N,C,T,V*window_size)
        N, C, T, V = x.shape
        x = self.unfold(x)
        # Permute extra channels from window size to the graph dimension;
        # -1 for number of windows
        x = x.reshape(N, C, self.window_size, -1, V).permute(0, 1, 3, 2,
                                                             4).contiguous()
        x = x.reshape(N, C, -1, self.window_size * V)
        return x


class ST_MSGCN(BaseModule):

    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 num_scales,
                 window_size,
                 residual=False,
                 dropout=0,
                 act_cfg=dict(type='ReLU')):

        super().__init__()
        self.num_scales = num_scales
        self.window_size = window_size
        A = self.build_st_graph(A, window_size)

        A_scales = [
            k_adjacency(A, k, with_self=True) for k in range(num_scales)
        ]
        A_scales = np.stack([normalize_digraph(g) for g in A_scales])

        self.register_buffer('A', torch.Tensor(A_scales))
        self.V = len(A)

        self.PA = nn.Parameter(self.A.clone())
        nn.init.uniform_(self.PA, -1e-6, 1e-6)

        self.mlp = MLP(
            in_channels * num_scales, [out_channels],
            dropout=dropout,
            act_cfg=act_cfg)

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels):
            self.residual = lambda x: x
        else:
            self.residual = MLP(in_channels, [out_channels], act_cfg=None)

        self.act = build_activation_layer(act_cfg)

    def build_st_graph(self, A, window_size):
        if not isinstance(A, np.ndarray):
            A = A.data.cpu().numpy()

        assert len(A.shape) == 2 and A.shape[0] == A.shape[1]
        V = len(A)
        A_with_I = A + np.eye(V, dtype=A.dtype)

        A_large = np.tile(A_with_I, (window_size, window_size)).copy()
        return A_large

    def forward(self, x):
        N, C, T, V = x.shape  # T = number of windows, V = self.V * window_size
        A = self.A + self.PA

        # Perform Graph Convolution
        res = self.residual(x)
        agg = torch.einsum('kvu,nctv->nkctu', A, x)
        agg = agg.reshape(N, self.num_scales * C, T, V)
        out = self.mlp(agg)
        if res == 0:
            return self.act(out)
        else:
            return self.act(out + res)


class MSG3DBlock(BaseModule):

    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 num_scales,
                 window_size,
                 window_stride,
                 window_dilation,
                 embed_factor=1,
                 activation='relu'):

        super().__init__()
        self.window_size = window_size
        self.out_channels = out_channels
        self.embed_channels_in = out_channels // embed_factor
        self.embed_channels_out = out_channels // embed_factor
        if embed_factor == 1:
            self.in1x1 = nn.Identity()
            self.embed_channels_in = self.embed_channels_out = in_channels
            # The first STGC block changes channels right away;
            # others change at collapse
            if in_channels == 3:
                self.embed_channels_out = out_channels
        else:
            self.in1x1 = MLP(in_channels, [self.embed_channels_in])

        self.gcn3d = Sequential(
            UnfoldTemporalWindows(window_size, window_stride, window_dilation),
            ST_MSGCN(
                in_channels=self.embed_channels_in,
                out_channels=self.embed_channels_out,
                A=A,
                num_scales=num_scales,
                window_size=window_size))

        self.out_conv = nn.Conv3d(
            self.embed_channels_out,
            out_channels,
            kernel_size=(1, self.window_size, 1))
        self.out_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        N, _, T, V = x.shape
        x = self.in1x1(x)
        # Construct temporal windows and apply MS-GCN
        x = self.gcn3d(x)

        # Collapse the window dimension
        x = x.reshape(N, self.embed_channels_out, -1, self.window_size, V)
        x = self.out_conv(x).squeeze(dim=3)
        x = self.out_bn(x)
        # no activation
        return x


class MW_MSG3DBlock(BaseModule):

    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 num_scales,
                 window_sizes=[3, 5],
                 window_stride=1,
                 window_dilations=[1, 1]):

        super().__init__()
        self.gcn3d = ModuleList([
            MSG3DBlock(in_channels, out_channels, A, num_scales, window_size,
                       window_stride, window_dilation) for window_size,
            window_dilation in zip(window_sizes, window_dilations)
        ])

    def forward(self, x):
        out_sum = 0
        for gcn3d in self.gcn3d:
            out_sum += gcn3d(x)
        return out_sum
