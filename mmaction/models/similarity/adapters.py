# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from mmengine.model import BaseModule

from mmaction.registry import MODELS


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call."""
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform quick gelu."""
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    """"ResidualAttentionBlock.

    Args:
        d_model (int): The dimension of the model.
        n_head (int): The number of heads.
        attn_mask (torch.Tensor, optional): The attention mask.
            Defaults to None.
    """

    def __init__(self,
                 d_model: int,
                 n_head: int,
                 attn_mask: Optional[torch.Tensor] = None) -> None:
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict([('c_fc', nn.Linear(d_model, d_model * 4)),
                         ('gelu', QuickGELU()),
                         ('c_proj', nn.Linear(d_model * 4, d_model))]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        """Perform attention."""
        self.attn_mask = self.attn_mask.to(
            dtype=x.dtype,
            device=x.device) if self.attn_mask is not None else None
        return self.attn(
            x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call."""
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    """"ResidualAttentionBlock.

    Args:
        width (int): The width of transformer.
        heads (int): The number of heads of transformer.
        layers (int): The number of layers of transformer.
        attn_mask (torch.Tensor, optional): The attention mask.
            Defaults to None.
    """

    def __init__(self,
                 width: int,
                 layers: int,
                 heads: int,
                 attn_mask: Optional[torch.Tensor] = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[
            ResidualAttentionBlock(width, heads, attn_mask)
            for _ in range(layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call."""
        return self.resblocks(x)


@MODELS.register_module()
class TransformerAdapter(BaseModule):
    """"Transformer adapter, modified from github.com/openai/CLIP.

    Args:
        num_segs (int): The number of segments.
        transformer_width (int): The width of transformer.
        transformer_heads (int): The number of heads of transformer.
        transformer_layers (int): The number of layers of transformer.
    """

    def __init__(self, num_segs: int, transformer_width: int,
                 transformer_heads: int, transformer_layers: int) -> None:
        super(TransformerAdapter, self).__init__()
        self.num_segs = num_segs

        self.positional_embedding = nn.Parameter(
            torch.empty(num_segs, transformer_width))
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads)

    def init_weights(self) -> None:
        """Initialize the weights."""

        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width**-0.5) * (
            (2 * self.transformer.layers)**-0.5)
        attn_std = self.transformer.width**-0.5
        fc_std = (2 * self.transformer.width)**-0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call."""
        b, seq_length, c = x.size()

        x_original = x
        x = x + self.positional_embedding
        x = x.transpose(0, 1)  # NLD -> LND
        x = self.transformer(x)
        x = x.transpose(0, 1)  # LND -> NLD
        x = x.type(x_original.dtype) + x_original
        return x.mean(dim=1)


@MODELS.register_module()
class SimpleMeanAdapter(BaseModule):
    """Average features adapter.

    Args:
        dim (int): The dimension to perform averaging. Defaults to 1.
    """

    def __init__(self, dim: Union[int, Tuple[int]] = 1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call."""
        return x.mean(dim=self.dim)
