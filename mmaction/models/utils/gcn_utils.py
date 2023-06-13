# Copyright (c) OpenMMLab. All rights reserved.
import copy as cp
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmengine.model import BaseModule, ModuleList, Sequential


class unit_gcn(BaseModule):
    """The basic unit of graph convolutional network.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        A (torch.Tensor): The adjacency matrix defined in the graph
            with shape of `(num_subsets, num_nodes, num_nodes)`.
        adaptive (str): The strategy for adapting the weights of the
            adjacency matrix. Defaults to ``'importance'``.
        conv_pos (str): The position of the 1x1 2D conv.
            Defaults to ``'pre'``.
        with_res (bool): Whether to use residual connection.
            Defaults to False.
        norm (str): The name of norm layer. Defaults to ``'BN'``.
        act (str): The name of activation layer. Defaults to ``'Relu'``.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 A: torch.Tensor,
                 adaptive: str = 'importance',
                 conv_pos: str = 'pre',
                 with_res: bool = False,
                 norm: str = 'BN',
                 act: str = 'ReLU',
                 init_cfg: Optional[Union[Dict, List[Dict]]] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_subsets = A.size(0)

        assert adaptive in [None, 'init', 'offset', 'importance']
        self.adaptive = adaptive
        assert conv_pos in ['pre', 'post']
        self.conv_pos = conv_pos
        self.with_res = with_res

        self.norm_cfg = norm if isinstance(norm, dict) else dict(type=norm)
        self.act_cfg = act if isinstance(act, dict) else dict(type=act)
        self.bn = build_norm_layer(self.norm_cfg, out_channels)[1]
        self.act = build_activation_layer(self.act_cfg)

        if self.adaptive == 'init':
            self.A = nn.Parameter(A.clone())
        else:
            self.register_buffer('A', A)

        if self.adaptive in ['offset', 'importance']:
            self.PA = nn.Parameter(A.clone())
            if self.adaptive == 'offset':
                nn.init.uniform_(self.PA, -1e-6, 1e-6)
            elif self.adaptive == 'importance':
                nn.init.constant_(self.PA, 1)

        if self.conv_pos == 'pre':
            self.conv = nn.Conv2d(in_channels, out_channels * A.size(0), 1)
        elif self.conv_pos == 'post':
            self.conv = nn.Conv2d(A.size(0) * in_channels, out_channels, 1)

        if self.with_res:
            if in_channels != out_channels:
                self.down = Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    build_norm_layer(self.norm_cfg, out_channels)[1])
            else:
                self.down = lambda x: x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call."""
        n, c, t, v = x.shape
        res = self.down(x) if self.with_res else 0

        A_switch = {None: self.A, 'init': self.A}
        if hasattr(self, 'PA'):
            A_switch.update({
                'offset': self.A + self.PA,
                'importance': self.A * self.PA
            })
        A = A_switch[self.adaptive]

        if self.conv_pos == 'pre':
            x = self.conv(x)
            x = x.view(n, self.num_subsets, -1, t, v)
            x = torch.einsum('nkctv,kvw->nctw', (x, A)).contiguous()
        elif self.conv_pos == 'post':
            x = torch.einsum('nctv,kvw->nkctw', (x, A)).contiguous()
            x = x.view(n, -1, t, v)
            x = self.conv(x)

        return self.act(self.bn(x) + res)


class unit_aagcn(BaseModule):
    """The graph convolution unit of AAGCN.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        A (torch.Tensor): The adjacency matrix defined in the graph
            with shape of `(num_subsets, num_joints, num_joints)`.
        coff_embedding (int): The coefficient for downscaling the embedding
            dimension. Defaults to 4.
        adaptive (bool): Whether to use adaptive graph convolutional layer.
            Defaults to True.
        attention (bool): Whether to use the STC-attention module.
            Defaults to True.
        init_cfg (dict or list[dict]): Initialization config dict. Defaults to
            ``[
                dict(type='Constant', layer='BatchNorm2d', val=1,
                     override=dict(type='Constant', name='bn', val=1e-6)),
                dict(type='Kaiming', layer='Conv2d', mode='fan_out'),
                dict(type='ConvBranch', name='conv_d')
            ]``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A: torch.Tensor,
        coff_embedding: int = 4,
        adaptive: bool = True,
        attention: bool = True,
        init_cfg: Optional[Union[Dict, List[Dict]]] = [
            dict(
                type='Constant',
                layer='BatchNorm2d',
                val=1,
                override=dict(type='Constant', name='bn', val=1e-6)),
            dict(type='Kaiming', layer='Conv2d', mode='fan_out'),
            dict(type='ConvBranch', name='conv_d')
        ]
    ) -> None:

        if attention:
            attention_init_cfg = [
                dict(
                    type='Constant',
                    layer='Conv1d',
                    val=0,
                    override=dict(type='Xavier', name='conv_sa')),
                dict(
                    type='Kaiming',
                    layer='Linear',
                    mode='fan_in',
                    override=dict(type='Constant', val=0, name='fc2c'))
            ]
            init_cfg = cp.copy(init_cfg)
            init_cfg.extend(attention_init_cfg)

        super(unit_aagcn, self).__init__(init_cfg=init_cfg)
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.num_subset = A.shape[0]
        self.adaptive = adaptive
        self.attention = attention

        num_joints = A.shape[-1]

        self.conv_d = ModuleList()
        for i in range(self.num_subset):
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if self.adaptive:
            self.A = nn.Parameter(A)

            self.alpha = nn.Parameter(torch.zeros(1))
            self.conv_a = ModuleList()
            self.conv_b = ModuleList()
            for i in range(self.num_subset):
                self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
                self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
        else:
            self.register_buffer('A', A)

        if self.attention:
            self.conv_ta = nn.Conv1d(out_channels, 1, 9, padding=4)
            # s attention
            ker_joint = num_joints if num_joints % 2 else num_joints - 1
            pad = (ker_joint - 1) // 2
            self.conv_sa = nn.Conv1d(out_channels, 1, ker_joint, padding=pad)
            # channel attention
            rr = 2
            self.fc1c = nn.Linear(out_channels, out_channels // rr)
            self.fc2c = nn.Linear(out_channels // rr, out_channels)

        self.down = lambda x: x
        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels))

        self.bn = nn.BatchNorm2d(out_channels)
        self.tan = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call."""
        N, C, T, V = x.size()

        y = None
        if self.adaptive:
            for i in range(self.num_subset):
                A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(
                    N, V, self.inter_c * T)
                A2 = self.conv_b[i](x).view(N, self.inter_c * T, V)
                A1 = self.tan(torch.matmul(A1, A2) / A1.size(-1))  # N V V
                A1 = self.A[i] + A1 * self.alpha
                A2 = x.view(N, C * T, V)
                z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
                y = z + y if y is not None else z
        else:
            for i in range(self.num_subset):
                A1 = self.A[i]
                A2 = x.view(N, C * T, V)
                z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
                y = z + y if y is not None else z

        y = self.relu(self.bn(y) + self.down(x))

        if self.attention:
            # spatial attention first
            se = y.mean(-2)  # N C V
            se1 = self.sigmoid(self.conv_sa(se))  # N 1 V
            y = y * se1.unsqueeze(-2) + y
            # then temporal attention
            se = y.mean(-1)  # N C T
            se1 = self.sigmoid(self.conv_ta(se))  # N 1 T
            y = y * se1.unsqueeze(-1) + y
            # then spatial temporal attention ??
            se = y.mean(-1).mean(-1)  # N C
            se1 = self.relu(self.fc1c(se))
            se2 = self.sigmoid(self.fc2c(se1))  # N C
            y = y * se2.unsqueeze(-1).unsqueeze(-1) + y
            # A little bit weird
        return y


class unit_tcn(BaseModule):
    """The basic unit of temporal convolutional network.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the temporal convolution kernel.
            Defaults to 9.
        stride (int): Stride of the temporal convolution. Defaults to 1.
        dilation (int): Spacing between temporal kernel elements.
            Defaults to 1.
        norm (str): The name of norm layer. Defaults to ``'BN'``.
        dropout (float): Dropout probability. Defaults to 0.
        init_cfg (dict or list[dict]): Initialization config dict. Defaults to
            ``[
                dict(type='Constant', layer='BatchNorm2d', val=1),
                dict(type='Kaiming', layer='Conv2d', mode='fan_out')
            ]``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 9,
        stride: int = 1,
        dilation: int = 1,
        norm: str = 'BN',
        dropout: float = 0,
        init_cfg: Union[Dict, List[Dict]] = [
            dict(type='Constant', layer='BatchNorm2d', val=1),
            dict(type='Kaiming', layer='Conv2d', mode='fan_out')
        ]
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_cfg = norm if isinstance(norm, dict) else dict(type=norm)
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))
        self.bn = build_norm_layer(self.norm_cfg, out_channels)[1] \
            if norm is not None else nn.Identity()

        self.drop = nn.Dropout(dropout, inplace=True)
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call."""
        return self.drop(self.bn(self.conv(x)))


class mstcn(BaseModule):
    """The multi-scale temporal convolutional network.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        mid_channels (int): Number of middle channels. Defaults to None.
        dropout (float): Dropout probability. Defaults to 0.
        ms_cfg (list): The config of multi-scale branches. Defaults to
            ``[(3, 1), (3, 2), (3, 3), (3, 4), ('max', 3), '1x1']``.
        stride (int): Stride of the temporal convolution. Defaults to 1.
        init_cfg (dict or list[dict]): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 mid_channels: int = None,
                 dropout: float = 0.,
                 ms_cfg: List = [(3, 1), (3, 2), (3, 3), (3, 4), ('max', 3),
                                 '1x1'],
                 stride: int = 1,
                 init_cfg: Union[Dict, List[Dict]] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        # Multiple branches of temporal convolution
        self.ms_cfg = ms_cfg
        num_branches = len(ms_cfg)
        self.num_branches = num_branches
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = nn.ReLU()

        if mid_channels is None:
            mid_channels = out_channels // num_branches
            rem_mid_channels = out_channels - mid_channels * (num_branches - 1)
        else:
            assert isinstance(mid_channels, float) and mid_channels > 0
            mid_channels = int(out_channels * mid_channels)
            rem_mid_channels = mid_channels

        self.mid_channels = mid_channels
        self.rem_mid_channels = rem_mid_channels

        branches = []
        for i, cfg in enumerate(ms_cfg):
            branch_c = rem_mid_channels if i == 0 else mid_channels
            if cfg == '1x1':
                branches.append(
                    nn.Conv2d(
                        in_channels,
                        branch_c,
                        kernel_size=1,
                        stride=(stride, 1)))
                continue
            assert isinstance(cfg, tuple)
            if cfg[0] == 'max':
                branches.append(
                    Sequential(
                        nn.Conv2d(in_channels, branch_c, kernel_size=1),
                        nn.BatchNorm2d(branch_c), self.act,
                        nn.MaxPool2d(
                            kernel_size=(cfg[1], 1),
                            stride=(stride, 1),
                            padding=(1, 0))))
                continue
            assert isinstance(cfg[0], int) and isinstance(cfg[1], int)
            branch = Sequential(
                nn.Conv2d(in_channels, branch_c, kernel_size=1),
                nn.BatchNorm2d(branch_c), self.act,
                unit_tcn(
                    branch_c,
                    branch_c,
                    kernel_size=cfg[0],
                    stride=stride,
                    dilation=cfg[1],
                    norm=None))
            branches.append(branch)

        self.branches = ModuleList(branches)
        tin_channels = mid_channels * (num_branches - 1) + rem_mid_channels

        self.transform = Sequential(
            nn.BatchNorm2d(tin_channels), self.act,
            nn.Conv2d(tin_channels, out_channels, kernel_size=1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout(dropout, inplace=True)

    def inner_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call."""
        N, C, T, V = x.shape

        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        feat = torch.cat(branch_outs, dim=1)
        feat = self.transform(feat)
        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call."""
        out = self.inner_forward(x)
        out = self.bn(out)
        return self.drop(out)
