# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import torch
from torch import Tensor, nn


def conv_block(in_channels: int,
               out_channels: int,
               kernel_size: int = 3,
               stride: int = 1) -> nn.Module:
    module = nn.Sequential(
        nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            bias=False), nn.BatchNorm1d(out_channels), nn.ReLU())
    return module


class Backbone(nn.Module):

    def __init__(self, channels_list: List[tuple]) -> None:
        super(Backbone, self).__init__()

        self.num_layers = len(channels_list)
        layers = []
        for idx, channels_config in enumerate(channels_list):
            layer = conv_block(*channels_config)
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

    def forward(self, x: Tensor, query_fts: Tensor,
                position_fts: Tensor) -> Tuple[Tensor]:
        results = []

        for idx in range(self.num_layers):
            query_ft = query_fts[idx].unsqueeze(1).permute(0, 2, 1)
            position_ft = position_fts[idx]
            x = query_ft * x
            if idx == 0:
                x = torch.cat([x, position_ft], dim=1)
            x = self.layers[idx](x)
            results.append(x)

        return tuple(results)
