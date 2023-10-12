# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import torch.nn.functional as F
from torch import Tensor, nn

from .backbone import conv_block


class FPN(nn.Module):

    def __init__(self, in_channels_list: List, out_channels: int) -> None:
        super(FPN, self).__init__()

        inner_blocks = []
        layer_blocks = []
        for idx, in_channels in enumerate(in_channels_list, 1):
            inner_block = conv_block(in_channels, out_channels, 1, 1)
            layer_block = conv_block(out_channels, out_channels, 3, 1)

            inner_blocks.append(inner_block)
            layer_blocks.append(layer_block)

        self.inner_blocks = nn.ModuleList(inner_blocks)
        self.layer_blocks = nn.ModuleList(layer_blocks)

    def forward(self, x: Tensor) -> Tuple[Tensor]:
        # process the last lowest resolution feat and
        # first feed it into 1 x 1 conv
        last_inner = self.inner_blocks[-1](x[-1])
        results = [self.layer_blocks[-1](last_inner)]

        for feature, inner_block, layer_block in zip(
                x[:-1][::-1], self.inner_blocks[:-1][::-1],
                self.layer_blocks[:-1][::-1]):
            if not inner_block:
                continue
            inner_top_down = F.interpolate(
                last_inner, scale_factor=2, mode='nearest')
            inner_lateral = inner_block(feature)
            last_inner = inner_lateral + inner_top_down
            results.insert(0, layer_block(last_inner))

        return tuple(results)
