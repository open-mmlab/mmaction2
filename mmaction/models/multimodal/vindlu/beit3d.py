# Copyright (c) OpenMMLab. All rights reserved.
import importlib
from typing import Dict, Optional, Tuple, Union

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.beit import BeitConfig, BeitModel
from transformers.models.beit.modeling_beit import BeitAttention, BeitDropPath
from transformers.models.beit.modeling_beit import \
    BeitEmbeddings as BeitEmbeddings2D
from transformers.models.beit.modeling_beit import BeitLayer as BeitLayer2D
from transformers.models.beit.modeling_beit import BeitRelativePositionBias
from transformers.models.beit.modeling_beit import \
    BeitRelativePositionBias as BeitRelativePositionBias2D

from mmaction.registry import MODELS
from .temporal_model import (X_CLIP, STAdapter, TemporalAttention,
                             WindowTemporalAttention)


def interpolate_temporal_pos_embed(temp_embed_old, num_frames_new):
    """
    temp_embed_old: (1, num_frames_old, 1, d)
    Returns:
        temp_embed_new: (1, num_frames_new, 1, d)
    """
    temp_embed_old = temp_embed_old.squeeze(2).permute(
        0, 2, 1)  # (1, d, num_frames_old)
    temp_embed_new = F.interpolate(
        temp_embed_old, num_frames_new,
        mode='linear')  # (1, d, num_frames_new)
    temp_embed_new = temp_embed_new.permute(0, 2, 1).unsqueeze(
        2)  # (1, num_frames_new, 1, d)
    return temp_embed_new


class TemporalAttentionBeit(nn.Module):
    """temporal attention using BeitAttention."""

    def __init__(self, config: BeitConfig):
        """TODO: to be defined."""
        super().__init__()

        self.layernorm_before = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.attention = BeitAttention(config, window_size=None)
        self.scale = nn.Parameter(
            config.temporal_model_init_value * torch.ones(
                (config.hidden_size)),
            requires_grad=True,
        )
        self.drop_path = BeitDropPath(config.drop_path_rate)

    def forward(self, hidden_states: torch.Tensor):
        """forward function.

        Args:
            hidden_states (torch.Tensor): The input. Shape: [b,t,l,c]

        Returns: TODO
        """
        b = hidden_states.shape[0]
        output = einops.rearrange(hidden_states, 'b t l c -> (b l) t c')
        output = self.layernorm_before(output)
        output = self.attention(output)
        output = einops.rearrange(output[0], '(b l) t c -> b t l c', b=b)
        return hidden_states + self.drop_path(output[0]) * self.scale


class BeitPooler3D(nn.Module):

    def __init__(self, config: BeitConfig) -> None:
        super().__init__()
        self.num_prompts = config.add_k_prompts
        self.layernorm = (
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            if config.use_mean_pooling else None)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (torch.Tensor): Shape: [B,T,L,C]
        """
        if self.layernorm is not None:
            # Mean pool the final hidden states of the patch tokens
            # patch_tokens = hidden_states[:, 1 + self.num_prompts :, :]
            if self.num_prompts > 0:
                patch_tokens = hidden_states[:, :, 1:-self.num_prompts, :]
            else:
                patch_tokens = hidden_states[:, :, 1:, :]
            pooled_output = self.layernorm(patch_tokens.mean(2))
        else:
            # Pool by simply taking the final hidden state of the [CLS] token
            pooled_output = hidden_states[:, :, 0]

        return pooled_output


class BeitRelativePositionBias3D(BeitRelativePositionBias2D):

    def __init__(self, config: BeitConfig, window_size: tuple) -> None:
        super().__init__(config, window_size)

        # add bias for prompts
        self.k = config.add_k_prompts
        if self.k > 0:
            self.prompt_bias_table = nn.parameter.Parameter(
                torch.zeros((2 + self.k) * self.k, config.num_attention_heads)
            )  # k prompt-to-token, k token-to-prompt, k*k prompt-to-promt
        else:
            self.prompt_bias_table = None

    def forward(self) -> torch.Tensor:
        # relative position bias 2d
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1] + 1,
                self.window_size[0] * self.window_size[1] + 1,
                -1,
            )  # Wh*Ww,Wh*Ww,nH

        # add bias for prompts
        k = self.k
        if k > 0:
            l = self.window_size[0] * self.window_size[1] + 1  # noqa: E741
            bias = torch.zeros(l + k, l + k,
                               relative_position_bias.shape[-1]).to(
                                   relative_position_bias.device)
            bias[:l, :l] = relative_position_bias
            bias[l:, :l] = self.prompt_bias_table[:k].view(
                k, 1, -1)  # prompt to token
            bias[:l,
                 l:] = self.prompt_bias_table[k:2 *
                                              k].view(1, k,
                                                      -1)  # token to prompt
            bias[l:, l:] = self.prompt_bias_table[2 * k, :].view(
                k, k, -1)  # prompt to prompt
        else:
            bias = relative_position_bias

        return bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww


class BeitEmbeddings3D(BeitEmbeddings2D):
    """Construct the CLS token, position and patch embeddings.

    Optionally, also the mask token.
    """

    def __init__(self, config: BeitConfig) -> None:
        super().__init__(config)

        if config.use_temporal_position_embedding:
            self.temporal_position_embeddings = nn.parameter.Parameter(
                torch.zeros(1, config.num_frames, 1, config.hidden_size))
        else:
            self.temporal_position_embeddings = None

        if config.add_k_prompts > 0:
            self.prompt_tokens = nn.parameter.Parameter(
                torch.zeros(1, config.add_k_prompts, config.hidden_size))
        else:
            self.prompt_tokens = None

    def forward(self,
                pixel_values: torch.Tensor,
                bool_masked_pos: Optional[torch.BoolTensor] = None
                ) -> torch.Tensor:
        """
        Args:
            pixel_values (torch.Tensor): The input image patches.
                Shape: [B, T, C, H, W].


        """
        t = pixel_values.shape[1]
        pixel_values = einops.rearrange(pixel_values,
                                        'b t c h w -> (b t) c h w')

        embeddings = self.patch_embeddings(pixel_values)
        batch_size, seq_len, _ = embeddings.size()  # [(b t) l c]

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        if bool_masked_pos is not None:
            mask_tokens = self.mask_token.expand(batch_size, seq_len, -1)
            # replace the masked visual tokens by mask_tokens
            w = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1 - w) + mask_tokens * w

        if self.prompt_tokens is not None:
            prompt_tokens = self.prompt_tokens.expand(batch_size, -1, -1)
            embeddings = torch.cat((cls_tokens, embeddings, prompt_tokens),
                                   dim=1)
        else:
            embeddings = torch.cat((cls_tokens, embeddings),
                                   dim=1)  # [B*T, L, C]
        if self.position_embeddings is not None:
            embeddings = embeddings + self.position_embeddings

        embeddings = einops.rearrange(embeddings, '(b t) l c -> b t l c', t=t)
        if self.temporal_position_embeddings is not None:
            if t <= self.temporal_position_embeddings.shape[1]:
                embeddings = embeddings + \
                    self.temporal_position_embeddings[:, :t]
            else:
                tpe = interpolate_temporal_pos_embed(
                    self.temporal_position_embeddings, t)
                embeddings = embeddings + tpe

        embeddings = self.dropout(embeddings)

        return embeddings


class BeitLayer3D(BeitLayer2D):

    def __init__(self,
                 config: BeitConfig,
                 window_size: Optional[tuple] = None,
                 drop_path_rate: float = 0.0) -> None:
        super().__init__(config, window_size, drop_path_rate)

        self.temporal_model_position = config.temporal_model_position
        if config.temporal_model_block == 'st_adapter':
            self.temp_model = STAdapter(**config.temporal_model_config)
        elif config.temporal_model_block == 'timesformer':
            self.temp_model = TemporalAttention(**config.temporal_model_config)
        elif config.temporal_model_block == 'ta_beit':
            self.temp_model = TemporalAttentionBeit(config)
        elif config.temporal_model_block == 'window_attention':
            self.temp_model = WindowTemporalAttention(
                **config.temporal_model_config)
        elif config.temporal_model_block == 'xclip':
            self.temp_model = X_CLIP(**config.temporal_model_config)
        elif config.temporal_model_block == 'none':
            self.temp_model = None
        else:
            raise ValueError(
                f'not accepted temporal model: {config.temporal_model_block}')

        self.temporal_model_block = config.temporal_model_block

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        relative_position_bias: Optional['BeitRelativePositionBias'] = None,
    ) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:

        b, t, l, c = hidden_states.shape

        if self.temporal_model_block == 'xclip':
            assert (self.temporal_model_position == 'first'
                    and self.config.add_k_prompts
                    == 1), ('xclip must be put before the attention and'
                            'add_k_prompts must be 1.')

        if self.temp_model is not None and \
           self.temporal_model_position == 'first':
            hidden_states = self.temp_model(hidden_states)

        hidden_states = einops.rearrange(hidden_states, 'b t l c -> (b t) l c')

        self_attention_outputs = self.attention(
            self.layernorm_before(
                hidden_states
            ),  # in BEiT, layernorm is applied before self-attention
            head_mask,
            output_attentions=output_attentions,
            relative_position_bias=relative_position_bias,
        )
        attention_output = self_attention_outputs[0]

        # add self attentions if we output attention weights
        outputs = self_attention_outputs[1:]

        # apply lambda_1 if present
        if self.lambda_1 is not None:
            attention_output = self.lambda_1 * attention_output

        # first residual connection
        hidden_states = self.drop_path(attention_output) + hidden_states

        # in BEiT, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)

        layer_output = self.intermediate(layer_output)
        layer_output = self.output(layer_output)

        if self.lambda_2 is not None:
            layer_output = self.lambda_2 * layer_output

        # second residual connection
        layer_output = self.drop_path(layer_output) + hidden_states

        layer_output = einops.rearrange(
            layer_output, '(b t) l c -> b t l c', b=b)

        # apply temporal modeling block
        if self.temp_model is not None and \
           self.temporal_model_position == 'last':
            layer_output = self.temp_model(layer_output)

        outputs = (layer_output, ) + outputs

        return outputs


class BeitConfig3D(BeitConfig):

    def __init__(self,
                 num_frames=1,
                 temporal_model_block='none',
                 temporal_model_position='last',
                 temporal_model_init_value=0.0,
                 temporal_model_config={},
                 use_temporal_position_embedding=False,
                 add_k_prompts=0,
                 **kwargs) -> None:

        super().__init__(**kwargs)
        self.temporal_model_block = temporal_model_block
        self.temporal_model_config = temporal_model_config
        self.temporal_model_position = temporal_model_position
        self.temporal_model_init_value = temporal_model_init_value
        self.use_temporal_position_embedding = use_temporal_position_embedding
        self.add_k_prompts = add_k_prompts
        self.num_frames = num_frames


@MODELS.register_module()
class BeitModel3D(BeitModel):

    def __init__(self,
                 config: BeitConfig,
                 tem_config: Dict,
                 add_pooling_layer: bool = True) -> None:
        # hack to replace original 2D modules with 3D modules
        beit_package = importlib.import_module(
            'transformers.models.beit.modeling_beit')
        beit_package.BeitEmbeddings = BeitEmbeddings3D
        beit_package.BeitPooler = BeitPooler3D
        beit_package.BeitLayer = BeitLayer3D
        beit_package.BeitRelativePositionBias = BeitRelativePositionBias3D

        config = BeitConfig3D.from_pretrained(config, **tem_config)
        super().__init__(config, add_pooling_layer)
