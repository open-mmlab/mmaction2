import torch
import torch.nn as nn
from mmcv.runner import load_checkpoint

from ...utils import get_root_logger
from ..registry import BACKBONES
from .resnet import ResNet


def temporal_shift(data, offset, temporal_size):
    """Shift data by different offsets according to their group.

    This operation is to shift a batch of frames in time dimension
    by a given offsets, which is range from [1, temporal_size].

    Args:
        data (torch.Tensor): Feature map data in shape
            [N, num_segments + 2, C, H, W].
        offset (torch.Tensor): Data offset used for shifting in shape
            [N, num_segments].
        temporal_size (int): Number of temporal size.
            [num_segments + 2].

    Returns:
        torch.Tensor: Result tensor in shape [N, num_segments, C, H, W].
    """
    # [N, num_segments + 2, C, H, W]
    n = data.shape[0]

    # [N]
    idx = torch.arange(n, dtype=torch.long, device=offset.device)
    # [N, num_segments + 2]
    idx = idx.view(-1, 1).expand(-1, temporal_size)

    # [N, num_segments, C, H, W]
    result = data[idx, offset, :]
    return result


def linear_sampler(data,
                   offset,
                   weight,
                   offset_weight=1.0,
                   extended=True,
                   scale=None):
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
        weight (torch.Tensor): Weight factors for this group data in shape
            [N, num_segments].
        offset_weight (float): Offset Weight. Default: 1.0.
        extended (bool): Whether to apply Temporal Extension. Default: True.
        scale (float | None): Scale factor for location. Default: None.
    """
    # [N, num_segments, C, H, W]
    n, t, c, h, w = data.shape

    location = torch.arange(1, t + 1, dtype=torch.float, device=data.device)
    location = location.view(1, -1).expand(n, -1)
    # [N, num_segments]

    if scale is not None:
        mean_location = t / 2.0 + 1
        location = (location - mean_location) * scale
        location = location + mean_location

    offset = offset * offset_weight + location
    offset0 = torch.floor(offset).long()
    offset1 = offset0 + 1
    # [N, num_segments]

    if extended:
        offset0 = torch.clamp(offset0, min=.0, max=t + 1)
        offset1 = torch.clamp(offset1, min=.0, max=t + 1)
    else:
        offset0 = torch.clamp(offset0, min=1, max=t)
        offset1 = torch.clamp(offset1, min=1, max=t)

    weight = weight[:, :, None, None, None]
    # [N, num_segments, 1, 1, 1]
    padding_data = torch.zeros((n, 1, c, h, w), device=data.device)
    # [N, 1, C, H, W]
    data = torch.cat([padding_data, data, padding_data], 1)
    # [N, num_segments + 2, C, H, W]

    data0 = temporal_shift(data, offset0, t)
    data1 = temporal_shift(data, offset1, t)
    # [N, num_segments, C, H, W]

    w0 = 1 - (offset - offset0.float())
    w1 = 1 - w0
    # [N, num_segments]

    w0 = w0[:, :, None, None, None]
    w1 = w1[:, :, None, None, None]
    # [N, num_segments, 1, 1, 1]

    output = w0 * data0 + w1 * data1
    output = output * weight
    # [N, num_segments, C, H, W]
    return output


class CombineNet(nn.Module):
    """Combine Net.

    It combines Temporal interlace module with some part of ResNet layer.

    Args:
        net1 (nn.module): Temporal interlace module.
        net2 (nn.module): Some part of resnet layer.
    """

    def __init__(self, net1, net2):
        super().__init__()
        self.net1 = net1
        self.net2 = net2
        self.buffer = []

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The output of the module.
        """
        # input shape: [num_batches * num_segments, C, H, W]
        # output x shape: [num_batches * num_segments, C, H, W]
        # x_offset shape: [num_batches, 1, groups]
        # w_weight shape: [num_batches, num_segmnets, groups]
        x, x_offset, x_weight = self.net1(x)
        # [num_batches * num_segments, C, H, W]
        x = self.net2(x)
        self.buffer.clear()
        self.buffer.append(x_offset.cpu().detach().numpy())
        self.buffer.append(x_weight.cpu().detach().numpy())
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
        # hard code kernel_size and padding according to original repo.
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

    This module is proposed in "Temporal Interlacing Network".
    Link: https://arxiv.org/abs/2001.06499

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
        # [num_batches * num_segments, C, H, W]
        n, c, h, w = x.size()
        num_batches = n // self.num_segments
        num_folds = c // self.shift_div

        # [num_batches * num_segments, C - num_folds, H, W]
        x_unchanged = x[:, num_folds:, :, :]
        # [num_batches * num_segments, num_folds, H, W]
        x_descriptor = x[:, :num_folds, :, :]
        # [num_batches, num_segments, num_folds, H, W]
        x_descriptor = x_descriptor.view(num_batches, self.num_segments,
                                         num_folds, h, w)

        # x should only obtain information on temporal and channel dimensions
        # [num_batches, num_segments, num_folds, W]
        x_pooled = torch.mean(x_descriptor, 3)
        # [num_batches, num_segments, num_folds]
        x_pooled = torch.mean(x_pooled, 3)
        # [num_batches, num_folds, num_segments]
        x_pooled = x_pooled.permute(0, 2, 1).contiguous()

        # Calculate weight and bias
        # [num_batches, 1, groups]
        x_offset = self.offset_net(x_pooled)
        # [num_batches, 1, groups]
        x_weight = self.weight_net(x_pooled)
        x_offset = -x_offset

        fold_interval = num_folds // self.deform_groups // 2
        # [num_batches, num_segments, num_folds, H, W]
        # parrots doesn't support zeors_like(x, device=x.device)
        x_shift = torch.zeros(x_descriptor.shape, device=x_descriptor.device)
        for i in range(self.deform_groups):
            x_shift[:, :, i * fold_interval:(i + 1) *
                    fold_interval, :, :] = linear_sampler(
                        x_descriptor[:, :, i * fold_interval:(i + 1) *
                                     fold_interval, :, :], x_offset[:, :, i],
                        x_weight[:, :, i])
        for i in range(self.deform_groups):
            p = i + self.deform_groups
            x_shift[:, :, p * fold_interval:(p + 1) *
                    fold_interval, :, :] = linear_sampler(
                        x_descriptor[:, :, p * fold_interval:(p + 1) *
                                     fold_interval, :, :], -x_offset[:, :, i],
                        x_weight[:, :, i])

        # [num_batches * num_segments, num_folds, H, W]
        x_shift = x_shift.contiguous().view(n, num_folds, h, w)
        # [num_batches * num_segments, C, H, W]
        x_out = torch.cat([x_shift, x_unchanged], dim=1)
        # output x shape: [num_batches * num_segments, C, H, W]
        # x_offset shape: [num_batches, 1, groups]
        # w_weight shape: [num_batches, num_segmnets, groups]
        return x_out, x_offset, x_weight


@BACKBONES.register_module()
class ResNetTIN(ResNet):
    """ResNet backbone for TIN.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        num_segments (int): Number of frame segments. Default: 8.
        finetune (str | None): Name of finetune model. Default: None.
        is_tin (bool): Whether to apply temporal interlace. Default: True.
        shift_div (int): Number of division parts for shift. Default: 4.
        temporal_pool (bool): Whether to add temporal pooling. Default: False.
        kwargs (dict, optional): Arguments for ResNet.
    """

    def __init__(self,
                 depth,
                 num_segments=8,
                 finetune=None,
                 is_dtn=True,
                 shift_div=4,
                 temporal_pool=False,
                 **kwargs):
        super().__init__(depth, **kwargs)
        self.num_segments = num_segments
        self.finetune = finetune
        self.is_dtn = is_dtn
        self.shift_div = shift_div
        self.temporal_pool = temporal_pool

    def make_temporal_interlace(self):
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
                    blocks[i].conv1 = CombineNet(tds, blocks[i].conv1)
            return nn.Sequential(*blocks)

        self.layer1 = make_block_interlace(
            self.layer1, num_segment_list[0], shift_div=self.shift_div)
        self.layer2 = make_block_interlace(
            self.layer2, num_segment_list[1], shift_div=self.shift_div)
        self.layer3 = make_block_interlace(
            self.layer3, num_segment_list[2], shift_div=self.shift_div)
        self.layer4 = make_block_interlace(
            self.layer4, num_segment_list[3], shift_div=self.shift_div)

    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        super().init_weights()
        if self.is_dtn:
            self.make_temporal_interlace()
        if self.finetune is not None:
            if isinstance(self.finetune, str):
                logger = get_root_logger()
                load_checkpoint(
                    self, self.finetune, strict=False, logger=logger)
            else:
                raise TypeError('finetune must be a str or None.')
