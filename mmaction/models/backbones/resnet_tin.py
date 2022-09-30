# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from mmaction.registry import MODELS
from .resnet_tsm import ResNetTSM


def linear_sampler(data, offset):
    """Differentiable Temporal-wise Frame Sampling, which is essentially a
    linear interpolation process.

    It gets the feature map which has been split into several groups
    and shift them by different offsets according to their groups.
    Then compute the weighted sum along with the temporal dimension.

    Args:
        data (torch.Tensor): Split data for certain group in shape
            [N, num_segments, C, H, W].
        offset (torch.Tensor): Data offsets for this group data in shape
            [N, num_segments].
    """
    # [N, num_segments, C, H, W]
    n, t, c, h, w = data.shape

    # offset0, offset1: [N, num_segments]
    offset0 = torch.floor(offset).int()
    offset1 = offset0 + 1

    # data, data0, data1: [N, num_segments, C, H * W]
    data = data.view(n, t, c, h * w).contiguous()

    try:
        from mmcv.ops import tin_shift
    except (ImportError, ModuleNotFoundError):
        raise ImportError('Failed to import `tin_shift` from `mmcv.ops`. You '
                          'will be unable to use TIN. ')

    data0 = tin_shift(data, offset0)
    data1 = tin_shift(data, offset1)

    # weight0, weight1: [N, num_segments]
    weight0 = 1 - (offset - offset0.float())
    weight1 = 1 - weight0

    # weight0, weight1:
    # [N, num_segments] -> [N, num_segments, C // num_segments] -> [N, C]
    group_size = offset.shape[1]
    weight0 = weight0[:, :, None].repeat(1, 1, c // group_size)
    weight0 = weight0.view(weight0.size(0), -1)
    weight1 = weight1[:, :, None].repeat(1, 1, c // group_size)
    weight1 = weight1.view(weight1.size(0), -1)

    # weight0, weight1: [N, C] -> [N, 1, C, 1]
    weight0 = weight0[:, None, :, None]
    weight1 = weight1[:, None, :, None]

    # output: [N, num_segments, C, H * W] -> [N, num_segments, C, H, W]
    output = weight0 * data0 + weight1 * data1
    output = output.view(n, t, c, h, w)

    return output


class CombineNet(nn.Module):
    """Combine Net.

    It combines Temporal interlace module with some part of ResNet layer.

    Args:
        net1 (nn.module): Temporal interlace module.
        net2 (nn.module): Some part of ResNet layer.
    """

    def __init__(self, net1, net2):
        super().__init__()
        self.net1 = net1
        self.net2 = net2

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The output of the module.
        """
        # input shape: [num_batches * num_segments, C, H, W]
        # output x shape: [num_batches * num_segments, C, H, W]
        x = self.net1(x)
        # [num_batches * num_segments, C, H, W]
        x = self.net2(x)
        return x


class WeightNet(nn.Module):
    """WeightNet in Temporal interlace module.

    The WeightNet consists of two parts: one convolution layer
    and a sigmoid function. Following the convolution layer, the sigmoid
    function and rescale module can scale our output to the range (0, 2).
    Here we set the initial bias of the convolution layer to 0, and the
    final initial output will be 1.0.

    Args:
        in_channels (int): Channel num of input features.
        groups (int): Number of groups for fc layer outputs.
    """

    def __init__(self, in_channels, groups):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.groups = groups

        self.conv = nn.Conv1d(in_channels, groups, 3, padding=1)

        self.init_weights()

    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        # we set the initial bias of the convolution
        # layer to 0, and the final initial output will be 1.0
        self.conv.bias.data[...] = 0

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The output of the module.
        """
        # calculate weight
        # [N, C, T]
        n, _, t = x.shape
        # [N, groups, T]
        x = self.conv(x)
        x = x.view(n, self.groups, t)
        # [N, T, groups]
        x = x.permute(0, 2, 1)

        # scale the output to range (0, 2)
        x = 2 * self.sigmoid(x)
        # [N, T, groups]
        return x


class OffsetNet(nn.Module):
    """OffsetNet in Temporal interlace module.

    The OffsetNet consists of one convolution layer and two fc layers
    with a relu activation following with a sigmoid function. Following
    the convolution layer, two fc layers and relu are applied to the output.
    Then, apply the sigmoid function with a multiply factor and a minus 0.5
    to transform the output to (-4, 4).

    Args:
        in_channels (int): Channel num of input features.
        groups (int): Number of groups for fc layer outputs.
        num_segments (int): Number of frame segments.
    """

    def __init__(self, in_channels, groups, num_segments):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        # hard code ``kernel_size`` and ``padding`` according to original repo.
        kernel_size = 3
        padding = 1

        self.conv = nn.Conv1d(in_channels, 1, kernel_size, padding=padding)
        self.fc1 = nn.Linear(num_segments, num_segments)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(num_segments, groups)

        self.init_weights()

    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        # The bias of the last fc layer is initialized to
        # make the post-sigmoid output start from 1
        self.fc2.bias.data[...] = 0.5108

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The output of the module.
        """
        # calculate offset
        # [N, C, T]
        n, _, t = x.shape
        # [N, 1, T]
        x = self.conv(x)
        # [N, T]
        x = x.view(n, t)
        # [N, T]
        x = self.relu(self.fc1(x))
        # [N, groups]
        x = self.fc2(x)
        # [N, 1, groups]
        x = x.view(n, 1, -1)

        # to make sure the output is in (-t/2, t/2)
        # where t = num_segments = 8
        x = 4 * (self.sigmoid(x) - 0.5)
        # [N, 1, groups]
        return x


class TemporalInterlace(nn.Module):
    """Temporal interlace module.

    This module is proposed in `Temporal Interlacing Network
    <https://arxiv.org/abs/2001.06499>`_

    Args:
        in_channels (int): Channel num of input features.
        num_segments (int): Number of frame segments. Default: 3.
        shift_div (int): Number of division parts for shift. Default: 1.
    """

    def __init__(self, in_channels, num_segments=3, shift_div=1):
        super().__init__()
        self.num_segments = num_segments
        self.shift_div = shift_div
        self.in_channels = in_channels
        # hard code ``deform_groups`` according to original repo.
        self.deform_groups = 2

        self.offset_net = OffsetNet(in_channels // shift_div,
                                    self.deform_groups, num_segments)
        self.weight_net = WeightNet(in_channels // shift_div,
                                    self.deform_groups)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The output of the module.
        """
        # x: [N, C, H, W],
        # where N = num_batches x num_segments, C = shift_div * num_folds
        n, c, h, w = x.size()
        num_batches = n // self.num_segments
        num_folds = c // self.shift_div

        # x_out: [num_batches x num_segments, C, H, W]
        x_out = torch.zeros((n, c, h, w), device=x.device)
        # x_descriptor: [num_batches, num_segments, num_folds, H, W]
        x_descriptor = x[:, :num_folds, :, :].view(num_batches,
                                                   self.num_segments,
                                                   num_folds, h, w)

        # x should only obtain information on temporal and channel dimensions
        # x_pooled: [num_batches, num_segments, num_folds, W]
        x_pooled = torch.mean(x_descriptor, 3)
        # x_pooled: [num_batches, num_segments, num_folds]
        x_pooled = torch.mean(x_pooled, 3)
        # x_pooled: [num_batches, num_folds, num_segments]
        x_pooled = x_pooled.permute(0, 2, 1).contiguous()

        # Calculate weight and bias, here groups = 2
        # x_offset: [num_batches, groups]
        x_offset = self.offset_net(x_pooled).view(num_batches, -1)
        # x_weight: [num_batches, num_segments, groups]
        x_weight = self.weight_net(x_pooled)

        # x_offset: [num_batches, 2 * groups]
        x_offset = torch.cat([x_offset, -x_offset], 1)
        # x_shift: [num_batches, num_segments, num_folds, H, W]
        x_shift = linear_sampler(x_descriptor, x_offset)

        # x_weight: [num_batches, num_segments, groups, 1]
        x_weight = x_weight[:, :, :, None]
        # x_weight:
        # [num_batches, num_segments, groups * 2, c // self.shift_div // 4]
        x_weight = x_weight.repeat(1, 1, 2, num_folds // 2 // 2)
        # x_weight:
        # [num_batches, num_segments, c // self.shift_div = num_folds]
        x_weight = x_weight.view(x_weight.size(0), x_weight.size(1), -1)

        # x_weight: [num_batches, num_segments, num_folds, 1, 1]
        x_weight = x_weight[:, :, :, None, None]
        # x_shift: [num_batches, num_segments, num_folds, H, W]
        x_shift = x_shift * x_weight
        # x_shift: [num_batches, num_segments, num_folds, H, W]
        x_shift = x_shift.contiguous().view(n, num_folds, h, w)

        # x_out: [num_batches x num_segments, C, H, W]
        x_out[:, :num_folds, :] = x_shift
        x_out[:, num_folds:, :] = x[:, num_folds:, :]

        return x_out


@MODELS.register_module()
class ResNetTIN(ResNetTSM):
    """ResNet backbone for TIN.

    Args:
        depth (int): Depth of ResNet, from {18, 34, 50, 101, 152}.
        num_segments (int): Number of frame segments. Default: 8.
        is_tin (bool): Whether to apply temporal interlace. Default: True.
        shift_div (int): Number of division parts for shift. Default: 4.
        kwargs (dict, optional): Arguments for ResNet.
    """

    def __init__(self, depth, is_tin=True, **kwargs):
        self.is_tin = is_tin
        super().__init__(depth, **kwargs)

    def init_structure(self):
        if self.is_tin:
            self.make_temporal_interlace()
        if len(self.non_local_cfg) != 0:
            self.make_non_local()

    def make_temporal_interlace(self):
        """Make temporal interlace for some layers."""
        num_segment_list = [self.num_segments] * 4
        assert num_segment_list[-1] > 0

        n_round = 1
        if len(list(self.layer3.children())) >= 23:
            print(f'=> Using n_round {n_round} to insert temporal shift.')

        def make_block_interlace(stage, num_segments, shift_div):
            """Apply Deformable shift for a ResNet layer module.

            Args:
                stage (nn.module): A ResNet layer to be deformed.
                num_segments (int): Number of frame segments.
                shift_div (int): Number of division parts for shift.

            Returns:
                nn.Sequential: A Sequential container consisted of
                    deformed Interlace blocks.
            """
            blocks = list(stage.children())
            for i, b in enumerate(blocks):
                if i % n_round == 0:
                    tds = TemporalInterlace(
                        b.conv1.in_channels,
                        num_segments=num_segments,
                        shift_div=shift_div)
                    blocks[i].conv1.conv = CombineNet(tds,
                                                      blocks[i].conv1.conv)
            return nn.Sequential(*blocks)

        self.layer1 = make_block_interlace(self.layer1, num_segment_list[0],
                                           self.shift_div)
        self.layer2 = make_block_interlace(self.layer2, num_segment_list[1],
                                           self.shift_div)
        self.layer3 = make_block_interlace(self.layer3, num_segment_list[2],
                                           self.shift_div)
        self.layer4 = make_block_interlace(self.layer4, num_segment_list[3],
                                           self.shift_div)

    def init_weights(self):
        pass
