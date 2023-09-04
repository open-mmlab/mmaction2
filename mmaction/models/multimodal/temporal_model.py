# Copyright (c) OpenMMLab. All rights reserved.
import math

import einops
import torch
from einops import rearrange
from timm.models.layers import DropPath
from torch import nn
from torch.nn import LayerNorm, Linear, MultiheadAttention


class STAdapter(nn.Module):
    """ST Adapter."""

    def __init__(
        self,
        kernel_size=(3, 3, 3),
        input_dim=768,
        hidden_dim=384,
        img_size=224,
        patch_size=16,
        drop_prob=0.1,
    ):
        super(STAdapter, self).__init__()
        self.kernel_size = kernel_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.h = self.w = img_size // patch_size

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, input_dim)
        self.act = nn.ReLU()
        self.conv = nn.Conv3d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            padding='same',
            groups=hidden_dim)
        self.droppath = DropPath(drop_prob=drop_prob)

        self.scale = nn.parameter.Parameter(torch.zeros([]))

    def forward(self, x: torch.Tensor):
        """forward.

        Args:
            x (torch.Tensor): input features.
            Shape: [bs, nframes, l, c]. l = 1 + h*w

        Returns: features after adapter. The same shape as input.
        """
        if x.shape[1] == 1:  # for single frame, return itself.
            return x

        shortcut = x
        x = self.linear1(x)
        cls = x[:, :, :1, :]
        tokens = x[:, :, 1:, :]
        tokens = einops.rearrange(
            tokens, 'b t (h w) c -> b c t h w', h=self.h).contiguous()
        tokens = self.conv(tokens)
        tokens = einops.rearrange(tokens, 'b c t h w -> b t (h w) c')
        x = torch.cat([cls, tokens], dim=2)  # [b, t, 1+h*w, c]
        x = self.act(x)
        x = self.linear2(x)

        return shortcut + self.scale * self.droppath(x)


class TemporalAttention(nn.Module):
    """perform temporal self-attention."""

    def __init__(self, input_dim=768, droppath_rate=0.1):
        """

        Kwargs:
            input_dim (int): The input feature dimension.


        """
        super().__init__()

        self._input_dim = input_dim
        self.temporal_attn = MultiheadAttention(
            input_dim, num_heads=input_dim // 64)
        self.norm = LayerNorm(input_dim, eps=1e-12)
        self.linear = Linear(input_dim, input_dim)
        self.droppath = DropPath(droppath_rate)
        self.scale = nn.parameter.Parameter(torch.zeros([]))

    def forward(self, x: torch.Tensor):
        """forward.

        Args:
            x (torch.Tensor): input features.
            Shape: [bs, nframes, l, c]. l = 1 + h*w

        Returns: features after adapter. The same shape as input.
        """
        if x.shape[1] == 1:  # for single frame, return itself.
            return x

        shortcut = x
        x = einops.rearrange(x, 'b t l c -> t (b l) c')
        x = self.norm(x)
        x = self.temporal_attn(x, x, x)[0]
        x = einops.rearrange(x, 't (b l) c -> b t l c', b=shortcut.shape[0])
        return shortcut + self.scale * self.droppath(x)


class WindowTemporalAttention(nn.Module):
    """perform windowed temporal self-attention."""

    def __init__(self, input_dim=768, droppath_rate=0.1, window_size=(2, 2)):
        """

        Kwargs:
            input_dim (int): The input feature dimension.


        """
        super().__init__()

        self._input_dim = input_dim
        self.temporal_attn = MultiheadAttention(
            input_dim, num_heads=input_dim // 64)
        self.norm = LayerNorm(input_dim, eps=1e-12)
        self.droppath = DropPath(droppath_rate)
        self.scale = nn.parameter.Parameter(torch.zeros([]))
        self.wh, self.ww = window_size

    def forward(self, x: torch.Tensor):
        """forward.

        Args:
            x (torch.Tensor): input features.
            Shape: [bs, nframes, l, c]. l = 1 + h*w

        Returns: features after adapter. The same shape as input.
        """
        if x.shape[1] == 1:  # for single frame, return itself.
            return x
        shortcut = x

        h = w = int(math.sqrt(x.shape[2] - 1))
        cls_token = x[:, :, :1, :]
        x = einops.rearrange(
            x[:, :, 1:, :],
            'b t (nh wh nw ww) c -> (t wh ww) (b nh nw) c',
            nh=h // self.wh,
            wh=self.wh,
            nw=w // self.ww,
            ww=self.ww,
        )
        x = self.norm(x)
        x = self.temporal_attn(x, x, x)[0]
        x = einops.rearrange(
            x,
            '(t wh ww) (b nh nw) c -> b t (nh wh nw ww) c',
            wh=self.wh,
            ww=self.ww,
            nh=h // self.wh,
            nw=w // self.ww,
        )
        # add back cls token.
        x = torch.concat([cls_token, x], dim=2)
        return shortcut + self.scale * self.droppath(x)


class X_CLIP(nn.Module):
    """perform windowed temporal self-attention."""

    def __init__(self, input_dim=768, droppath_rate=0.1, num_prompts=1):
        """

        Kwargs:
            input_dim (int): The input feature dimension.


        """
        super().__init__()

        d_model = input_dim

        self.message_fc = nn.Linear(d_model, d_model)
        self.message_ln = LayerNorm(d_model, eps=1e-12)
        self.message_attn = nn.MultiheadAttention(d_model, d_model // 64)
        self.num_prompts = num_prompts

        self.droppath = DropPath(droppath_rate)

    def forward(self, x: torch.Tensor):
        """forward.

        Args:
            x (torch.Tensor): input features.
            Shape: [bs, nframes, l, c]. l = 1 + h*w

        Returns: features after adapter. The same shape as input.
        """
        if x.shape[1] == 1:  # for single frame, return itself.
            return x
        msg_token = self.message_ln(self.message_fc(x[:, :,
                                                      0, :]))  # [b, t, c]
        msg_token = rearrange(msg_token, 'b t c -> t b c')
        msg_token = msg_token + self.droppath(
            self.message_attn(msg_token, msg_token, msg_token)[0])
        msg_token = rearrange(msg_token, 't b c -> b t c')
        # replace the last prompt token with msg_token.
        x = torch.cat([x[:, :, :-1, :],
                       msg_token.unsqueeze(2)], dim=2)  # [b, t, l+1, c]
        return x
