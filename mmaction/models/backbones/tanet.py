# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from typing import Optional

import torch
import torch.nn as nn
from torch.utils import checkpoint as cp

from mmaction.registry import MODELS
from ..common import TAM
from .resnet import Bottleneck, ResNet


class TABlock(nn.Module):
    """Temporal Adaptive Block (TA-Block) for TANet.

    This block is proposed in `TAM: TEMPORAL ADAPTIVE MODULE FOR VIDEO
    RECOGNITION <https://arxiv.org/pdf/2005.06803>`_

    The temporal adaptive module (TAM) is embedded into ResNet-Block
    after the first Conv2D, which turns the vanilla ResNet-Block
    into TA-Block.

    Args:
        block (nn.Module): Residual blocks to be substituted.
        num_segments (int): Number of frame segments.
        tam_cfg (dict): Config for temporal adaptive module (TAM).
    """

    def __init__(self, block: nn.Module, num_segments: int,
                 tam_cfg: dict) -> None:
        super().__init__()
        self.tam_cfg = deepcopy(tam_cfg)
        self.block = block
        self.num_segments = num_segments
        self.tam = TAM(
            in_channels=block.conv1.out_channels,
            num_segments=num_segments,
            **self.tam_cfg)

        if not isinstance(self.block, Bottleneck):
            raise NotImplementedError('TA-Blocks have not been fully '
                                      'implemented except the pattern based '
                                      'on Bottleneck block.')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call."""

        assert isinstance(self.block, Bottleneck)

        def _inner_forward(x):
            """Forward wrapper for utilizing checkpoint."""
            identity = x

            out = self.block.conv1(x)
            out = self.tam(out)
            out = self.block.conv2(out)
            out = self.block.conv3(out)

            if self.block.downsample is not None:
                identity = self.block.downsample(x)

            out = out + identity

            return out

        if self.block.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.block.relu(out)

        return out


@MODELS.register_module()
class TANet(ResNet):
    """Temporal Adaptive Network (TANet) backbone.

    This backbone is proposed in `TAM: TEMPORAL ADAPTIVE MODULE FOR VIDEO
    RECOGNITION <https://arxiv.org/pdf/2005.06803>`_

    Embedding the temporal adaptive module (TAM) into ResNet to
    instantiate TANet.

    Args:
        depth (int): Depth of resnet, from ``{18, 34, 50, 101, 152}``.
        num_segments (int): Number of frame segments.
        tam_cfg (dict, optional): Config for temporal adaptive module (TAM).
            Defaults to None.
    """

    def __init__(self,
                 depth: int,
                 num_segments: int,
                 tam_cfg: Optional[dict] = None,
                 **kwargs) -> None:
        super().__init__(depth, **kwargs)
        assert num_segments >= 3
        self.num_segments = num_segments
        tam_cfg = dict() if tam_cfg is None else tam_cfg
        self.tam_cfg = deepcopy(tam_cfg)
        super().init_weights()
        self.make_tam_modeling()

    def init_weights(self):
        pass

    def make_tam_modeling(self):
        """Replace ResNet-Block with TA-Block."""

        def make_tam_block(stage, num_segments, tam_cfg=dict()):
            blocks = list(stage.children())
            for i, block in enumerate(blocks):
                blocks[i] = TABlock(block, num_segments, deepcopy(tam_cfg))
            return nn.Sequential(*blocks)

        for i in range(self.num_stages):
            layer_name = f'layer{i + 1}'
            res_layer = getattr(self, layer_name)
            setattr(self, layer_name,
                    make_tam_block(res_layer, self.num_segments, self.tam_cfg))
