import torch
import torch.nn as nn

from mmaction.models.registry import BACKBONES
from .resnet import ResNet


class TemporalShift(nn.Module):
    """Temporal shift module.

    This module is proposed in
    "TSM: Temporal Shift Module for Efficient Video Understanding".
    Link: https://arxiv.org/abs/1811.08383

    Args:
        net (nn.module): Module to make temporal shift.
        num_segments (int): Number of frame segments. Default: 3.
        n_div (int): Number of div for shift. Default: 8.
    """

    def __init__(self, net, num_segments=3, n_div=8):
        super(TemporalShift, self).__init__()
        self.net = net
        self.num_segments = num_segments
        self.fold_div = n_div

    def forward(self, x):
        x = self.shift(x, self.num_segments, fold_div=self.fold_div)
        return self.net(x)

    @staticmethod
    def shift(x, num_segments, fold_div=3):
        # [N, C, H, W]
        n, c, h, w = x.size()
        # [N // num_segments, num_segments, C, H, W]
        x = x.view(-1, num_segments, c, h, w)

        fold = c // fold_div
        # [N // num_segments, num_segments, C, H, W]
        out = torch.zeros_like(x)
        out[:, :-1, :fold, :, :] = x[:, 1:, :fold, :, :]  # shift left
        out[:, 1:, fold:2 * fold, :, :] = x[:, :-1,
                                            fold:2 * fold, :, :]  # shift right
        out[:, :, 2 * fold:, :, :] = x[:, :, 2 * fold:, :, :]  # not shift

        # [N, C, H, W]
        return out.view(n, c, h, w)


@BACKBONES.register_module
class ResNetTSM(ResNet):
    """ResNet backbone for TSM.

    Args:
        num_segments (int): Number of frame segments. Default: 8.
        is_shift (bool): Whether to make temporal shift in reset layers.
            Default: True.
        shift_div (int): Number of div for shift. Default: 8.
        shift_place (str): Places in resnet layers for shift, which is chosen
            from ['block', 'blockres'].
            If set to 'block', it will apply temporal shift to all child blocks
                in each resnet layer.
            If set to 'blockres', it will apply temporal shift to each `conv1`
                layer of all child blocks in each resnet layer.
            Default: 'blockres'.
        temporal_pool (bool): Whether to add temporal pooling. Default: False.
        **kwargs (keyword arguments, optional): Arguments for ResNet.
    """

    def __init__(self,
                 depth,
                 num_segments=8,
                 is_shift=True,
                 shift_div=8,
                 shift_place='blockres',
                 temporal_pool=False,
                 **kwargs):
        super(ResNetTSM, self).__init__(depth, **kwargs)
        self.num_segments = num_segments
        self.is_shift = is_shift
        self.shift_div = shift_div
        self.shift_place = shift_place
        self.temporal_pool = temporal_pool

    def make_temporal_shift(self):
        if self.temporal_pool:
            num_segment_list = [
                self.num_segments, self.num_segments // 2,
                self.num_segments // 2, self.num_segments // 2
            ]
        else:
            num_segment_list = [self.num_segments] * 4
        if not num_segment_list[-1] > 0:
            raise ValueError('num_segment_list[-1] must be positive')

        if self.shift_place == 'block':

            def make_block_temporal(stage, num_segments):
                blocks = list(stage.children())
                for i, b in enumerate(blocks):
                    blocks[i] = TemporalShift(
                        b, num_segments=num_segments, n_div=self.shift_div)
                return nn.Sequential(*blocks)

            self.layer1 = make_block_temporal(self.layer1, num_segment_list[0])
            self.layer2 = make_block_temporal(self.layer2, num_segment_list[1])
            self.layer3 = make_block_temporal(self.layer3, num_segment_list[2])
            self.layer4 = make_block_temporal(self.layer4, num_segment_list[3])

        elif 'blockres' in self.shift_place:
            n_round = 1
            if len(list(self.layer3.children())) >= 23:
                n_round = 2

            def make_block_temporal(stage, num_segments):
                blocks = list(stage.children())
                for i, b in enumerate(blocks):
                    if i % n_round == 0:
                        blocks[i].conv1 = TemporalShift(
                            b.conv1,
                            num_segments=num_segments,
                            n_div=self.shift_div)
                return nn.Sequential(*blocks)

            self.layer1 = make_block_temporal(self.layer1, num_segment_list[0])
            self.layer2 = make_block_temporal(self.layer2, num_segment_list[1])
            self.layer3 = make_block_temporal(self.layer3, num_segment_list[2])
            self.layer4 = make_block_temporal(self.layer4, num_segment_list[3])

        else:
            raise NotImplementedError

    def init_weights(self):
        super(ResNetTSM, self).init_weights()
        if self.is_shift:
            self.make_temporal_shift()
