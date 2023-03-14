import torch
import torch.nn as nn
from mmcv.cnn import build_activation_layer
from mmengine.model import BaseModule, ModuleList, Sequential

from mmaction.models.utils import unit_tcn


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


class CTRGC(BaseModule):

    def __init__(self,
                 in_channels,
                 out_channels,
                 rel_reduction=8,
                 init_cfg=[
                     dict(type='Constant', layer='BatchNorm2d', val=1),
                     dict(type='Kaiming', layer='Conv2d', mode='fan_out')
                 ]):
        super(CTRGC, self).__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels <= 16:
            self.rel_channels = 8
        else:
            self.rel_channels = in_channels // rel_reduction
        self.conv1 = nn.Conv2d(
            self.in_channels, self.rel_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(
            self.in_channels, self.rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(
            self.in_channels, self.out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(
            self.rel_channels, self.out_channels, kernel_size=1)
        self.tanh = nn.Tanh()

    def forward(self, x, A=None, alpha=1):
        # Input: N, C, T, V
        x1, x2, x3 = self.conv1(x).mean(-2), self.conv2(x).mean(
            -2), self.conv3(x)
        # X1, X2: N, R, V
        # N, R, V, 1 - N, R, 1, V
        x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
        # N, R, V, V
        x1 = self.conv4(x1) * alpha + (A[None, None] if A is not None else 0
                                       )  # N,C,V,V
        x1 = torch.einsum('ncuv,nctu->nctv', x1, x3)
        return x1


class unit_ctrgcn(BaseModule):

    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 init_cfg=[
                     dict(
                         type='Constant',
                         layer='BatchNorm2d',
                         val=1,
                         override=dict(type='Constant', name='bn', val=1e-6)),
                     dict(type='Kaiming', layer='Conv2d', mode='fan_out')
                 ]):

        super(unit_ctrgcn, self).__init__(init_cfg=init_cfg)
        inter_channels = out_channels // 4
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels

        self.num_subset = A.shape[0]
        self.convs = ModuleList()

        for i in range(self.num_subset):
            self.convs.append(CTRGC(in_channels, out_channels))

        if in_channels != out_channels:
            self.down = Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels))
        else:
            self.down = lambda x: x

        self.A = nn.Parameter(A.clone())

        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = None

        for i in range(self.num_subset):
            z = self.convs[i](x, self.A[i], self.alpha)
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        return self.relu(y)
