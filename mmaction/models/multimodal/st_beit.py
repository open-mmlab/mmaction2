# Copyright 2021 Microsoft Research and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BEiT model."""

import collections.abc
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import einops
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import (BaseModelOutput,
                                           BaseModelOutputWithPooling,
                                           ImageClassifierOutput,
                                           MaskedLMOutput,
                                           SemanticSegmenterOutput)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import (find_pruneable_heads_and_indices,
                                        prune_linear_layer)
from transformers.utils import (add_code_sample_docstrings,
                                add_start_docstrings,
                                add_start_docstrings_to_model_forward, logging,
                                replace_return_docstrings)

from .temporal_model import (X_CLIP, STAdapter, TemporalAttention, TemporalS4,
                             WindowTemporalAttention)
from .utils import interpolate_temporal_pos_embed

logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = 'BeitConfig'
_FEAT_EXTRACTOR_FOR_DOC = 'BeitFeatureExtractor'

# Base docstring
_CHECKPOINT_FOR_DOC = 'microsoft/beit-base-patch16-224-pt22k'
_EXPECTED_OUTPUT_SHAPE = [1, 197, 768]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = 'microsoft/beit-base-patch16-224'
_IMAGE_CLASS_EXPECTED_OUTPUT = 'tabby, tabby cat'

BEIT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    'microsoft/beit-base-patch16-224',
    # See all BEiT models at https://huggingface.co/models?filter=beit
]


class BeitConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`BeitModel`]. It is used to instantiate an BEiT
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the BEiT
    [microsoft/beit-base-patch16-224-pt22k](https://huggingface.co/microsoft/beit-base-patch16-224-pt22k) architecture.

    Args:
        vocab_size (`int`, *optional*, defaults to 8092):
            Vocabulary size of the BEiT model. Defines the number of different image tokens that can be used during
            pre-training.
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        use_mask_token (`bool`, *optional*, defaults to `False`):
            Whether to use a mask token for masked image modeling.
        use_absolute_position_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to use BERT-style absolute position embeddings.
        use_relative_position_bias (`bool`, *optional*, defaults to `False`):
            Whether to use T5-style relative position embeddings in the self-attention layers.
        use_shared_relative_position_bias (`bool`, *optional*, defaults to `False`):
            Whether to use the same relative position embeddings across all self-attention layers of the Transformer.
        layer_scale_init_value (`float`, *optional*, defaults to 0.1):
            Scale to use in the self-attention layers. 0.1 for base, 1e-5 for large. Set 0 to disable layer scale.
        drop_path_rate (`float`, *optional*, defaults to 0.1):
            Stochastic depth rate per sample (when applied in the main path of residual layers).
        use_mean_pooling (`bool`, *optional*, defaults to `True`):
            Whether to mean pool the final hidden states of the patches instead of using the final hidden state of the
            CLS token, before applying the classification head.
        out_indices (`List[int]`, *optional*, defaults to `[3, 5, 7, 11]`):
            Indices of the feature maps to use for semantic segmentation.
        pool_scales (`Tuple[int]`, *optional*, defaults to `[1, 2, 3, 6]`):
            Pooling scales used in Pooling Pyramid Module applied on the last feature map.
        use_auxiliary_head (`bool`, *optional*, defaults to `True`):
            Whether to use an auxiliary head during training.
        auxiliary_loss_weight (`float`, *optional*, defaults to 0.4):
            Weight of the cross-entropy loss of the auxiliary head.
        auxiliary_channels (`int`, *optional*, defaults to 256):
            Number of channels to use in the auxiliary head.
        auxiliary_num_convs (`int`, *optional*, defaults to 1):
            Number of convolutional layers to use in the auxiliary head.
        auxiliary_concat_input (`bool`, *optional*, defaults to `False`):
            Whether to concatenate the output of the auxiliary head with the input before the classification layer.
        semantic_loss_ignore_index (`int`, *optional*, defaults to 255):
            The index that is ignored by the loss function of the semantic segmentation model.

    Example:

    ```python
    >>> from transformers import BeitModel, BeitConfig

    >>> # Initializing a BEiT beit-base-patch16-224-pt22k style configuration
    >>> configuration = BeitConfig()

    >>> # Initializing a model from the beit-base-patch16-224-pt22k style configuration
    >>> model = BeitModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = 'beit'

    def __init__(
        self,
        vocab_size=8192,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act='gelu',
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        is_encoder_decoder=False,
        image_size=224,
        num_frames=1,
        patch_size=16,
        num_channels=3,
        use_mask_token=False,
        use_absolute_position_embeddings=False,
        use_relative_position_bias=False,
        use_shared_relative_position_bias=False,
        layer_scale_init_value=0.1,
        drop_path_rate=0.1,
        use_mean_pooling=True,
        out_indices=[3, 5, 7, 11],
        pool_scales=[1, 2, 3, 6],
        use_auxiliary_head=True,
        auxiliary_loss_weight=0.4,
        auxiliary_channels=256,
        auxiliary_num_convs=1,
        auxiliary_concat_input=False,
        semantic_loss_ignore_index=255,
        temporal_model_block='none',
        temporal_model_position='last',
        temporal_model_init_value=0.0,
        temporal_model_config={},
        use_temporal_position_embedding=False,
        add_k_prompts=0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.use_mask_token = use_mask_token
        self.use_absolute_position_embeddings = use_absolute_position_embeddings
        self.use_relative_position_bias = use_relative_position_bias
        self.use_shared_relative_position_bias = use_shared_relative_position_bias
        self.layer_scale_init_value = layer_scale_init_value
        self.drop_path_rate = drop_path_rate
        self.use_mean_pooling = use_mean_pooling
        # decode head attributes (semantic segmentation)
        self.out_indices = out_indices
        self.pool_scales = pool_scales
        # auxiliary head attributes (semantic segmentation)
        self.use_auxiliary_head = use_auxiliary_head
        self.auxiliary_loss_weight = auxiliary_loss_weight
        self.auxiliary_channels = auxiliary_channels
        self.auxiliary_num_convs = auxiliary_num_convs
        self.auxiliary_concat_input = auxiliary_concat_input
        self.semantic_loss_ignore_index = semantic_loss_ignore_index

        self.temporal_model_block = temporal_model_block
        self.temporal_model_config = temporal_model_config
        self.temporal_model_position = temporal_model_position
        self.temporal_model_init_value = temporal_model_init_value
        self.use_temporal_position_embedding = use_temporal_position_embedding
        self.add_k_prompts = add_k_prompts
        self.num_frames = num_frames


@dataclass
class BeitModelOutputWithPooling(BaseModelOutputWithPooling):
    """Class for outputs of [`BeitModel`].

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            Average of the last layer hidden states of the patch tokens (excluding the *[CLS]* token) if
            *config.use_mean_pooling* is set to True. If set to False, then the final hidden state of the *[CLS]* token
            will be returned.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """


def drop_path(input: torch.Tensor,
              drop_prob: float = 0.0,
              training: bool = False) -> torch.Tensor:
    """Drop paths (Stochastic Depth) per sample (when applied in main path of
    residual blocks).

    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
    """
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0], ) + (1, ) * (
        input.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(
        shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # binarize
    output = input.div(keep_prob) * random_tensor
    return output


class BeitDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of
    residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


# Based on timm implementation, which can be found here:
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
class BeitEmbeddings(nn.Module):
    """Construct the CLS token, position and patch embeddings.

    Optionally, also the mask token.
    """

    def __init__(self, config: BeitConfig) -> None:
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        if config.use_mask_token:
            self.mask_token = nn.Parameter(
                torch.zeros(1, 1, config.hidden_size))
        else:
            self.mask_token = None
        self.patch_embeddings = BeitPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        if config.use_absolute_position_embeddings:
            self.position_embeddings = nn.Parameter(
                torch.zeros(1, num_patches + 1, config.hidden_size))
        else:
            self.position_embeddings = None
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

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
            pixel_values (torch.Tensor): The input image patches. Shape: [B, T, C, H, W].


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
                embeddings = embeddings + self.temporal_position_embeddings[:, :
                                                                            t]
            else:
                tpe = interpolate_temporal_pos_embed(
                    self.temporal_position_embeddings, t)
                embeddings = embeddings + tpe

        embeddings = self.dropout(embeddings)

        return embeddings


class BeitPatchEmbeddings(nn.Module):
    """This class turns `pixel_values` of shape `(batch_size, num_channels,
    height, width)` into the initial `hidden_states` (patch embeddings) of
    shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer."""

    def __init__(self, config):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size

        image_size = (
            image_size if isinstance(image_size, collections.abc.Iterable) else
            (image_size, image_size))
        patch_size = (
            patch_size if isinstance(patch_size, collections.abc.Iterable) else
            (patch_size, patch_size))
        num_patches = (image_size[1] // patch_size[1]) * (
            image_size[0] // patch_size[0])
        patch_shape = (image_size[0] // patch_size[0],
                       image_size[1] // patch_size[1])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.patch_shape = patch_shape

        self.projection = nn.Conv2d(
            num_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                'Make sure that the channel dimension of the pixel values match with the one set in the configuration.'
            )
        if height != self.image_size[0] or width != self.image_size[1]:
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
            )
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)

        return embeddings


class BeitSelfAttention(nn.Module):

    def __init__(self,
                 config: BeitConfig,
                 window_size: Optional[tuple] = None) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
                config, 'embedding_size'):
            raise ValueError(
                f'The hidden size {config.hidden_size,} is not a multiple of the number of attention '
                f'heads {config.num_attention_heads}.')

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size /
                                       config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(
            config.hidden_size, self.all_head_size, bias=False)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        if window_size:
            self.relative_position_bias = BeitRelativePositionBias(
                config, window_size=window_size)
        else:
            self.relative_position_bias = None

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads,
                                       self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        relative_position_bias: Optional['BeitRelativePositionBias'] = None,
    ) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer,
                                        key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(
            self.attention_head_size)

        # Add relative position bias if present.
        if self.relative_position_bias is not None:
            attention_scores = attention_scores + self.relative_position_bias(
            ).unsqueeze(0)

        # Add shared relative position bias if provided.
        if relative_position_bias is not None:
            attention_scores = attention_scores + relative_position_bias

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.all_head_size, )
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer,
                   attention_probs) if output_attentions else (context_layer, )

        return outputs


class BeitSelfOutput(nn.Module):
    """The residual connection is defined in BeitLayer instead of here (as is
    the case with other models), due to the layernorm applied before each
    block."""

    def __init__(self, config: BeitConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self,
                hidden_states: torch.Tensor,
                input_tensor: torch.Tensor,
                gamma=None) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class BeitAttention(nn.Module):

    def __init__(self,
                 config: BeitConfig,
                 window_size: Optional[tuple] = None) -> None:
        super().__init__()
        self.attention = BeitSelfAttention(config, window_size=window_size)
        self.output = BeitSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads,
            self.attention.num_attention_heads,
            self.attention.attention_head_size,
            self.pruned_heads,
        )

        # Prune linear layers
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(
            heads)
        self.attention.all_head_size = (
            self.attention.attention_head_size *
            self.attention.num_attention_heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        relative_position_bias: Optional['BeitRelativePositionBias'] = None,
    ) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        self_outputs = self.attention(hidden_states, head_mask,
                                      output_attentions,
                                      relative_position_bias)

        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,
                   ) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BeitIntermediate(nn.Module):

    def __init__(self, config: BeitConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states


class BeitOutput(nn.Module):

    def __init__(self, config: BeitConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


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


class BeitLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(
        self,
        config: BeitConfig,
        window_size: Optional[tuple] = None,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.config = config
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BeitAttention(config, window_size=window_size)
        self.intermediate = BeitIntermediate(config)
        self.output = BeitOutput(config)
        self.layernorm_before = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.drop_path = (
            BeitDropPath(drop_path_rate)
            if drop_path_rate > 0.0 else nn.Identity())
        self.layernorm_after = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)

        self.temporal_model_position = config.temporal_model_position

        init_values = config.layer_scale_init_value
        if init_values > 0:
            self.lambda_1 = nn.Parameter(
                init_values * torch.ones((config.hidden_size)),
                requires_grad=True)
            self.lambda_2 = nn.Parameter(
                init_values * torch.ones((config.hidden_size)),
                requires_grad=True)
        else:
            self.lambda_1, self.lambda_2 = None, None

        if config.temporal_model_block == 'st_adapter':
            self.temp_model = STAdapter(**config.temporal_model_config)
        elif config.temporal_model_block == 'timesformer':
            self.temp_model = TemporalAttention(**config.temporal_model_config)
        elif config.temporal_model_block == 's4':
            self.temp_model = TemporalS4(**config.temporal_model_config)
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
            assert (
                self.temporal_model_position == 'first'
                and self.config.add_k_prompts == 1
            ), 'xclip must be put before the attention and add_k_prompts must be 1.'

        if self.temp_model is not None and self.temporal_model_position == 'first':
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
        if self.temp_model is not None and self.temporal_model_position == 'last':
            layer_output = self.temp_model(layer_output)

        outputs = (layer_output, ) + outputs

        return outputs


class BeitRelativePositionBias(nn.Module):

    def __init__(self, config: BeitConfig, window_size: tuple) -> None:
        super().__init__()
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] -
                                      1) * (2 * window_size[1] - 1) + 3
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.num_relative_distance,
                        config.num_attention_heads))  # 2*Wh-1 * 2*Ww-1, nH
        # cls to token & token 2 cls & cls to cls

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = (coords_flatten[:, :, None] -
                           coords_flatten[:, None, :])  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = torch.zeros(
            size=(window_size[0] * window_size[1] + 1, ) * 2,
            dtype=relative_coords.dtype)
        relative_position_index[1:,
                                1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1

        self.register_buffer('relative_position_index',
                             relative_position_index)

        # add bias for prompts
        k = config.add_k_prompts
        self.k = k
        if k > 0:
            self.prompt_bias_table = nn.parameter.Parameter(
                torch.zeros((2 + k) * k, config.num_attention_heads)
            )  # k prompt-to-token, k token-to-prompt, k*k prompt-to-promt
        else:
            self.prompt_bias_table = None

    def forward(self) -> torch.Tensor:
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1] + 1,
                self.window_size[0] * self.window_size[1] + 1,
                -1,
            )  # Wh*Ww,Wh*Ww,nH

        k = self.k
        if k > 0:
            l = self.window_size[0] * self.window_size[1] + 1
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

            # bias[k:, k:] = relative_position_bias
            # bias[:k, k:] = self.prompt_bias_table[:k].view(k, 1, -1)
            # bias[k:, :k] = self.prompt_bias_table[k : 2 * k].view(1, k, -1)
            # bias[:k, :k] = self.prompt_bias_table[2 * k :].view(k, k, -1)
        else:
            bias = relative_position_bias

        return bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww


class BeitEncoder(nn.Module):

    def __init__(self,
                 config: BeitConfig,
                 window_size: Optional[tuple] = None) -> None:
        super().__init__()
        self.config = config
        if config.use_shared_relative_position_bias:
            self.relative_position_bias = BeitRelativePositionBias(
                config, window_size=window_size)
        else:
            self.relative_position_bias = None

        # stochastic depth decay rule
        dpr = [
            x.item() for x in torch.linspace(0, config.drop_path_rate,
                                             config.num_hidden_layers)
        ]
        self.layer = nn.ModuleList([
            BeitLayer(
                config,
                window_size=window_size
                if config.use_relative_position_bias else None,
                drop_path_rate=dpr[i],
            ) for i in range(config.num_hidden_layers)
        ])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                # all_hidden_states = all_hidden_states + (
                #     einops.rearrange(hidden_states, "b t l c -> (b t) l c"),
                # )
                all_hidden_states = all_hidden_states + (hidden_states, )

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):

                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    layer_head_mask,
                    use_reentrant=False,
                )
            else:
                relative_position_bias = (
                    self.relative_position_bias()
                    if self.relative_position_bias is not None else None)
                layer_outputs = layer_module(hidden_states, layer_head_mask,
                                             output_attentions,
                                             relative_position_bias)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (
                    layer_outputs[1], )

        # hidden_states = einops.rearrange(hidden_states, "b t l c -> (b t) l c")

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states, )

        if not return_dict:
            return tuple(
                v for v in
                [hidden_states, all_hidden_states, all_self_attentions]
                if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class BeitPreTrainedModel(PreTrainedModel):
    """An abstract class to handle weights initialization and a simple
    interface for downloading and loading pretrained models."""

    config_class = BeitConfig
    base_model_prefix = 'beit'
    main_input_name = 'pixel_values'
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, BeitEncoder):
            module.gradient_checkpointing = value


BEIT_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`BeitConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

BEIT_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`BeitFeatureExtractor`]. See
            [`BeitFeatureExtractor.__call__`] for details.

        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    'The bare Beit Model transformer outputting raw hidden-states without any specific head on top.',
    BEIT_START_DOCSTRING,
)
class BeitModel(BeitPreTrainedModel):

    def __init__(self,
                 config: BeitConfig,
                 add_pooling_layer: bool = True) -> None:
        super().__init__(config)
        self.config = config

        self.embeddings = BeitEmbeddings(config)
        self.encoder = BeitEncoder(
            config, window_size=self.embeddings.patch_embeddings.patch_shape)

        self.layernorm = (
            nn.Identity() if config.use_mean_pooling else nn.LayerNorm(
                config.hidden_size, eps=config.layer_norm_eps))
        self.pooler = BeitPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.

        heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(BEIT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_FEAT_EXTRACTOR_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BeitModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
        modality='vision',
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, BeitModelOutputWithPooling]:
        output_attentions = (
            output_attentions if output_attentions is not None else
            self.config.output_attentions)
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else
            self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError('You have to specify pixel_values')

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask,
                                       self.config.num_hidden_layers)

        # pixel_values: [bsz, nframes, c, h, w]
        assert pixel_values.ndim == 5, logger.error(
            f'input shape to st_beit: {pixel_values.shape}')

        embedding_output = self.embeddings(
            pixel_values, bool_masked_pos)  # [bs, nframes, L, c]

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(
            sequence_output) if self.pooler is not None else None

        # logger.info(f"sequence_output: {sequence_output.shape}. pooled_output: {pooled_output.shape}")

        if not return_dict:
            head_outputs = ((sequence_output,
                             pooled_output) if pooled_output is not None else
                            (sequence_output, ))
            return head_outputs + encoder_outputs[1:]

        return BeitModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class BeitPooler(nn.Module):

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


@add_start_docstrings(
    """Beit Model transformer with a 'language' modeling head on top. BEiT does masked image modeling by predicting
    visual tokens of a Vector-Quantize Variational Autoencoder (VQ-VAE), whereas other vision models like ViT and DeiT
    predict RGB pixel values. As a result, this class is incompatible with [`AutoModelForMaskedImageModeling`], so you
    will need to use [`BeitForMaskedImageModeling`] directly if you wish to do masked image modeling with BEiT.""",
    BEIT_START_DOCSTRING,
)
class BeitForMaskedImageModeling(BeitPreTrainedModel):

    def __init__(self, config: BeitConfig) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels
        self.beit = BeitModel(config, add_pooling_layer=False)

        # Classifier head
        self.layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(BEIT_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=MaskedLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, MaskedLMOutput]:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).

        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Returns:

        Examples:

        ```python
        >>> from transformers import BeitFeatureExtractor, BeitForMaskedImageModeling
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> feature_extractor = BeitFeatureExtractor.from_pretrained("microsoft/beit-base-patch16-224-pt22k")
        >>> model = BeitForMaskedImageModeling.from_pretrained("microsoft/beit-base-patch16-224-pt22k")

        >>> num_patches = (model.config.image_size // model.config.patch_size) ** 2
        >>> pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
        >>> # create random boolean mask of shape (batch_size, num_patches)
        >>> bool_masked_pos = torch.randint(low=0, high=2, size=(1, num_patches)).bool()

        >>> outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)
        >>> loss, logits = outputs.loss, outputs.logits
        >>> list(logits.shape)
        [1, 196, 8192]
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.beit(
            pixel_values,
            bool_masked_pos=bool_masked_pos,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        sequence_output = self.layernorm(sequence_output)
        prediction_scores = self.lm_head(sequence_output[:, 1:])

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores[bool_masked_pos],
                                      labels)

        if not return_dict:
            output = (prediction_scores, ) + outputs[1:]
            return ((masked_lm_loss, ) +
                    output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    Beit Model transformer with an image classification head on top (a linear layer on top of the average of the final
    hidden states of the patch tokens) e.g. for ImageNet.
    """,
    BEIT_START_DOCSTRING,
)
class BeitForImageClassification(BeitPreTrainedModel):

    def __init__(self, config: BeitConfig) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels
        self.beit = BeitModel(config, add_pooling_layer=True)

        # Classifier head
        self.classifier = (
            nn.Linear(config.hidden_size, config.num_labels)
            if config.num_labels > 0 else nn.Identity())

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(BEIT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_FEAT_EXTRACTOR_FOR_DOC,
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=ImageClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, ImageClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.beit(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs.pooler_output if return_dict else outputs[1]

        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = 'regression'
                elif self.num_labels > 1 and (labels.dtype == torch.long
                                              or labels.dtype == torch.int):
                    self.config.problem_type = 'single_label_classification'
                else:
                    self.config.problem_type = 'multi_label_classification'

            if self.config.problem_type == 'regression':
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == 'single_label_classification':
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == 'multi_label_classification':
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits, ) + outputs[2:]
            return ((loss, ) + output) if loss is not None else output

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BeitConvModule(nn.Module):
    """A convolutional block that bundles conv/norm/activation layers. This
    block simplifies the usage of convolution layers, which are commonly used
    with a norm layer (e.g., BatchNorm) and activation layer (e.g., ReLU).

    Based on OpenMMLab's implementation, found in https://github.com/open-
    mmlab/mmsegmentation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        padding: Union[int, Tuple[int, int], str] = 0,
        bias: bool = False,
        dilation: Union[int, Tuple[int, int]] = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
            dilation=dilation,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.conv(input)
        output = self.bn(output)
        output = self.activation(output)

        return output


class BeitPyramidPoolingBlock(nn.Module):

    def __init__(self, pool_scale: int, in_channels: int,
                 channels: int) -> None:
        super().__init__()
        self.layers = [
            nn.AdaptiveAvgPool2d(pool_scale),
            BeitConvModule(in_channels, channels, kernel_size=1),
        ]
        for i, layer in enumerate(self.layers):
            self.add_module(str(i), layer)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        hidden_state = input
        for layer in self.layers:
            hidden_state = layer(hidden_state)
        return hidden_state


class BeitPyramidPoolingModule(nn.Module):
    """Pyramid Pooling Module (PPM) used in PSPNet.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        align_corners (bool): align_corners argument of F.interpolate.

    Based on OpenMMLab's implementation, found in https://github.com/open-mmlab/mmsegmentation.
    """

    def __init__(
        self,
        pool_scales: Tuple[int, ...],
        in_channels: int,
        channels: int,
        align_corners: bool,
    ) -> None:
        super().__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels
        self.blocks = []
        for i, pool_scale in enumerate(pool_scales):
            block = BeitPyramidPoolingBlock(
                pool_scale=pool_scale,
                in_channels=in_channels,
                channels=channels)
            self.blocks.append(block)
            self.add_module(str(i), block)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        ppm_outs = []
        for ppm in self.blocks:
            ppm_out = ppm(x)
            upsampled_ppm_out = nn.functional.interpolate(
                ppm_out,
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs


class BeitUperHead(nn.Module):
    """Unified Perceptual Parsing for Scene Understanding. This head is the
    implementation of.

    [UPerNet](https://arxiv.org/abs/1807.10221).

    Based on OpenMMLab's implementation, found in https://github.com/open-mmlab/mmsegmentation.
    """

    def __init__(self, config: BeitConfig) -> None:
        super().__init__()

        self.pool_scales = config.pool_scales  # e.g. (1, 2, 3, 6)
        self.in_channels = [config.hidden_size
                            ] * 4  # e.g. [768, 768, 768, 768]
        self.channels = config.hidden_size
        self.align_corners = False
        self.classifier = nn.Conv2d(
            self.channels, config.num_labels, kernel_size=1)

        # PSP Module
        self.psp_modules = BeitPyramidPoolingModule(
            self.pool_scales,
            self.in_channels[-1],
            self.channels,
            align_corners=self.align_corners,
        )
        self.bottleneck = BeitConvModule(
            self.in_channels[-1] + len(self.pool_scales) * self.channels,
            self.channels,
            kernel_size=3,
            padding=1,
        )
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = BeitConvModule(in_channels, self.channels, kernel_size=1)
            fpn_conv = BeitConvModule(
                self.channels, self.channels, kernel_size=3, padding=1)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = BeitConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            kernel_size=3,
            padding=1,
        )

    def psp_forward(self, inputs):
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output

    def forward(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        # build laterals
        laterals = [
            lateral_conv(encoder_hidden_states[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp_forward(encoder_hidden_states))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + nn.functional.interpolate(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = nn.functional.interpolate(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners,
            )
        fpn_outs = torch.cat(fpn_outs, dim=1)
        output = self.fpn_bottleneck(fpn_outs)
        output = self.classifier(output)

        return output


class BeitFCNHead(nn.Module):
    """Fully Convolution Networks for Semantic Segmentation. This head is
    implemented of.

    [FCNNet](https://arxiv.org/abs/1411.4038>).

    Args:
        config (BeitConfig): Configuration.
        in_channels
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        dilation (int): The dilation rate for convs in the head. Default: 1.


    Based on OpenMMLab's implementation, found in https://github.com/open-mmlab/mmsegmentation.
    """

    def __init__(
        self,
        config: BeitConfig,
        in_index: int = 2,
        kernel_size: int = 3,
        dilation: Union[int, Tuple[int, int]] = 1,
    ) -> None:
        super().__init__()
        self.in_channels = config.hidden_size
        self.channels = config.auxiliary_channels
        self.num_convs = config.auxiliary_num_convs
        self.concat_input = config.auxiliary_concat_input
        self.in_index = in_index

        conv_padding = (kernel_size // 2) * dilation
        convs = []
        convs.append(
            BeitConvModule(
                self.in_channels,
                self.channels,
                kernel_size=kernel_size,
                padding=conv_padding,
                dilation=dilation,
            ))
        for i in range(self.num_convs - 1):
            convs.append(
                BeitConvModule(
                    self.channels,
                    self.channels,
                    kernel_size=kernel_size,
                    padding=conv_padding,
                    dilation=dilation,
                ))
        if self.num_convs == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)
        if self.concat_input:
            self.conv_cat = BeitConvModule(
                self.in_channels + self.channels,
                self.channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )

        self.classifier = nn.Conv2d(
            self.channels, config.num_labels, kernel_size=1)

    def forward(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        # just take the relevant feature maps
        hidden_states = encoder_hidden_states[self.in_index]
        output = self.convs(hidden_states)
        if self.concat_input:
            output = self.conv_cat(torch.cat([hidden_states, output], dim=1))
        output = self.classifier(output)
        return output


@add_start_docstrings(
    """
    Beit Model transformer with a semantic segmentation head on top e.g. for ADE20k, CityScapes.
    """,
    BEIT_START_DOCSTRING,
)
class BeitForSemanticSegmentation(BeitPreTrainedModel):

    def __init__(self, config: BeitConfig) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels
        self.beit = BeitModel(config, add_pooling_layer=False)

        # FPNs
        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(
                config.hidden_size,
                config.hidden_size,
                kernel_size=2,
                stride=2),
            nn.BatchNorm2d(config.hidden_size),
            nn.GELU(),
            nn.ConvTranspose2d(
                config.hidden_size,
                config.hidden_size,
                kernel_size=2,
                stride=2),
        )
        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(
                config.hidden_size,
                config.hidden_size,
                kernel_size=2,
                stride=2), )
        self.fpn3 = nn.Identity()
        self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Semantic segmentation head(s)
        self.decode_head = BeitUperHead(config)
        self.auxiliary_head = BeitFCNHead(
            config) if config.use_auxiliary_head else None

        # Initialize weights and apply final processing
        self.post_init()

    def compute_loss(self, logits, auxiliary_logits, labels):
        # upsample logits to the images' original size
        upsampled_logits = nn.functional.interpolate(
            logits,
            size=labels.shape[-2:],
            mode='bilinear',
            align_corners=False)
        if auxiliary_logits is not None:
            upsampled_auxiliary_logits = nn.functional.interpolate(
                auxiliary_logits,
                size=labels.shape[-2:],
                mode='bilinear',
                align_corners=False)
        # compute weighted loss
        loss_fct = CrossEntropyLoss(
            ignore_index=self.config.semantic_loss_ignore_index)
        main_loss = loss_fct(upsampled_logits, labels)
        auxiliary_loss = loss_fct(upsampled_auxiliary_logits, labels)
        loss = main_loss + self.config.auxiliary_loss_weight * auxiliary_loss

        return loss

    @add_start_docstrings_to_model_forward(BEIT_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=SemanticSegmenterOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, SemanticSegmenterOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            Ground truth semantic segmentation maps for computing the loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1`, a classification loss is computed (Cross-Entropy).

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoFeatureExtractor, BeitForSemanticSegmentation
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/beit-base-finetuned-ade-640-640")
        >>> model = BeitForSemanticSegmentation.from_pretrained("microsoft/beit-base-finetuned-ade-640-640")

        >>> inputs = feature_extractor(images=image, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> # logits are of shape (batch_size, num_labels, height, width)
        >>> logits = outputs.logits
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else
            self.config.output_hidden_states)

        outputs = self.beit(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=True,  # we need the intermediate hidden states
            return_dict=return_dict,
        )

        encoder_hidden_states = outputs.hidden_states if return_dict else outputs[
            1]

        # only keep certain features, and reshape
        # note that we do +1 as the encoder_hidden_states also includes the initial embeddings
        features = [
            feature for idx, feature in enumerate(encoder_hidden_states)
            if idx + 1 in self.config.out_indices
        ]
        batch_size = pixel_values.shape[0]
        patch_resolution = self.config.image_size // self.config.patch_size
        features = [
            x[:, 1:, :].permute(0, 2,
                                1).reshape(batch_size, -1, patch_resolution,
                                           patch_resolution) for x in features
        ]

        # apply FPNs
        ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
        for i in range(len(features)):
            features[i] = ops[i](features[i])

        logits = self.decode_head(features)

        auxiliary_logits = None
        if self.auxiliary_head is not None:
            auxiliary_logits = self.auxiliary_head(features)

        loss = None
        if labels is not None:
            if self.config.num_labels == 1:
                raise ValueError(
                    'The number of labels should be greater than one')
            else:
                loss = self.compute_loss(logits, auxiliary_logits, labels)

        if not return_dict:
            if output_hidden_states:
                output = (logits, ) + outputs[1:]
            else:
                output = (logits, ) + outputs[2:]
            return ((loss, ) + output) if loss is not None else output

        return SemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states
            if output_hidden_states else None,
            attentions=outputs.attentions,
        )
