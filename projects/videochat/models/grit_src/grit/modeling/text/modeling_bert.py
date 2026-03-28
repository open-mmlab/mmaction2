# Copyright 2018 The Google AI Language Team Authors and The HuggingFace
# Inc. team. Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""PyTorch BERT model."""
# Adapted from https://github.com/huggingface/transformers/blob/main/src
# /transformers/models/bert/modeling_bert.py

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import copy
import json
import logging
import math
import os
from io import open

import torch
import torch.utils.checkpoint as checkpoint
from torch import nn

from .file_utils import cached_path

logger = logging.getLogger()

BERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    'bert-base-uncased':
    'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased'
    '-config.json',
    'bert-large-uncased':
    'https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased'
    '-config.json',
    'bert-base-cased':
    'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased'
    '-config.json',
    'bert-large-cased':
    'https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased'
    '-config.json',
    'bert-base-multilingual-uncased':
    'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base'
    '-multilingual-uncased-config.json',
    'bert-base-multilingual-cased':
    'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base'
    '-multilingual-cased-config.json',
    'bert-base-chinese':
    'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese'
    '-config.json',
    'bert-base-german-cased':
    'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german'
    '-cased-config.json',
    'bert-large-uncased-whole-word-masking':
    'https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased'
    '-whole-word-masking-config.json',
    'bert-large-cased-whole-word-masking':
    'https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased'
    '-whole-word-masking-config.json',
    'bert-large-uncased-whole-word-masking-finetuned-squad':
    'https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased'
    '-whole-word-masking-finetuned-squad-config.json',
    'bert-large-cased-whole-word-masking-finetuned-squad':
    'https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased'
    '-whole-word-masking-finetuned-squad-config.json',
    'bert-base-cased-finetuned-mrpc':
    'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased'
    '-finetuned-mrpc-config.json',
}


def qk2attn(query, key, attention_mask, gamma):
    query = query / gamma
    attention_scores = torch.matmul(query, key.transpose(-1, -2))
    if attention_mask is not None:
        # Apply the attention mask is (precomputed for all layers in
        # BertModel forward() function)
        attention_scores = attention_scores + attention_mask
    return attention_scores.softmax(dim=-1)


class QK2Attention(nn.Module):

    def forward(self, query, key, attention_mask, gamma):
        return qk2attn(query, key, attention_mask, gamma)


LayerNormClass = torch.nn.LayerNorm


class BertSelfAttention(nn.Module):

    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                'The hidden size (%d) is not a multiple of the number of '
                'attention '
                'heads (%d)' %
                (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size /
                                       config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * \
            self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.softmax = nn.Softmax(dim=-1)
        self.qk2attn = QK2Attention()

    def transpose_for_scores(self, x):
        if torch._C._get_tracing_state():
            # exporter is not smart enough to detect dynamic size for some
            # paths
            x = x.view(x.shape[0], -1, self.num_attention_heads,
                       self.attention_head_size)
        else:
            new_x_shape = x.size()[:-1] + (self.num_attention_heads,
                                           self.attention_head_size)
            x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self,
                hidden_states,
                attention_mask,
                head_mask=None,
                history_state=None):
        if history_state is not None:
            x_states = torch.cat([history_state, hidden_states], dim=1)
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(x_states)
            mixed_value_layer = self.value(x_states)
        else:
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_probs = self.qk2attn(query_layer, key_layer, attention_mask,
                                       math.sqrt(self.attention_head_size))

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
                   attention_probs) if self.output_attentions else (
                       context_layer, )
        return outputs


class BertSelfOutput(nn.Module):

    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.pre_norm = hasattr(config, 'pre_norm') and config.pre_norm
        if not self.pre_norm:
            self.LayerNorm = LayerNormClass(
                config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if not self.pre_norm:
            hidden_states = self.LayerNorm(hidden_states + input_tensor)
        else:
            hidden_states = hidden_states + input_tensor
        return hidden_states


class BertAttention(nn.Module):

    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.pre_norm = hasattr(config, 'pre_norm') and config.pre_norm
        if self.pre_norm:
            self.LayerNorm = LayerNormClass(
                config.hidden_size, eps=config.layer_norm_eps)
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self,
                input_tensor,
                attention_mask,
                head_mask=None,
                history_state=None):
        if self.pre_norm:
            self_outputs = self.self(
                self.LayerNorm(input_tensor), attention_mask, head_mask,
                self.layerNorm(history_state)
                if history_state else history_state)
        else:
            self_outputs = self.self(input_tensor, attention_mask, head_mask,
                                     history_state)
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output,
                   ) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertIntermediate(nn.Module):

    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        assert config.hidden_act == 'gelu', \
            'Please implement other activation functions'
        self.intermediate_act_fn = _gelu_python

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):

    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.pre_norm = hasattr(config, 'pre_norm') and config.pre_norm
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if not self.pre_norm:
            self.LayerNorm = LayerNormClass(
                config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if not self.pre_norm:
            hidden_states = self.LayerNorm(hidden_states + input_tensor)
        else:
            hidden_states = hidden_states + input_tensor
        return hidden_states


class Mlp(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.pre_norm = hasattr(config, 'pre_norm') and config.pre_norm
        self.intermediate = BertIntermediate(config)
        if self.pre_norm:
            self.LayerNorm = LayerNormClass(
                config.hidden_size, eps=config.layer_norm_eps)
        self.output = BertOutput(config)

    def forward(self, attention_output):
        if not self.pre_norm:
            intermediate_output = self.intermediate(attention_output)
        else:
            intermediate_output = self.intermediate(
                self.LayerNorm(attention_output))
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertLayer(nn.Module):

    def __init__(self, config, use_act_checkpoint=True):
        super(BertLayer, self).__init__()
        self.pre_norm = hasattr(config, 'pre_norm') and config.pre_norm
        self.use_mlp_wrapper = hasattr(
            config, 'use_mlp_wrapper') and config.use_mlp_wrapper
        self.attention = BertAttention(config)
        self.use_act_checkpoint = use_act_checkpoint
        if self.use_mlp_wrapper:
            self.mlp = Mlp(config)
        else:
            self.intermediate = BertIntermediate(config)
            if self.pre_norm:
                self.LayerNorm = LayerNormClass(
                    config.hidden_size, eps=config.layer_norm_eps)
            self.output = BertOutput(config)

    def forward(self,
                hidden_states,
                attention_mask,
                head_mask=None,
                history_state=None):
        if self.use_act_checkpoint:
            attention_outputs = checkpoint.checkpoint(self.attention,
                                                      hidden_states,
                                                      attention_mask,
                                                      head_mask, history_state)
        else:
            attention_outputs = self.attention(hidden_states, attention_mask,
                                               head_mask, history_state)
        attention_output = attention_outputs[0]
        if self.use_mlp_wrapper:
            layer_output = self.mlp(attention_output)
        else:
            if not self.pre_norm:
                intermediate_output = self.intermediate(attention_output)
            else:
                intermediate_output = self.intermediate(
                    self.LayerNorm(attention_output))
            layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output, ) + attention_outputs[
            1:]  # add attentions if we output them
        return outputs


class BertEncoder(nn.Module):

    def __init__(self, config, use_act_checkpoint=True):
        super(BertEncoder, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([
            BertLayer(config, use_act_checkpoint=use_act_checkpoint)
            for _ in range(config.num_hidden_layers)
        ])
        self.pre_norm = hasattr(config, 'pre_norm') and config.pre_norm
        if self.pre_norm:
            self.LayerNorm = LayerNormClass(
                config.hidden_size, eps=config.layer_norm_eps)

    def forward(self,
                hidden_states,
                attention_mask,
                head_mask=None,
                encoder_history_states=None):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states, )

            history_state = None \
                if encoder_history_states is None \
                else encoder_history_states[i]
            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                (None if head_mask is None else head_mask[i]),
                history_state,
            )
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1], )
        if self.pre_norm:
            hidden_states = self.LayerNorm(hidden_states)
        outputs = (hidden_states, )
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states, )
        if self.output_attentions:
            outputs = outputs + (all_attentions, )
        return outputs


CONFIG_NAME = 'config.json'


class PretrainedConfig(object):
    """Base class for all configuration classes.

    Handle a few common parameters and methods for loading/downloading/saving
    configurations.
    """
    pretrained_config_archive_map = {}

    def __init__(self, **kwargs):
        self.finetuning_task = kwargs.pop('finetuning_task', None)
        self.num_labels = kwargs.pop('num_labels', 2)
        self.output_attentions = kwargs.pop('output_attentions', False)
        self.output_hidden_states = kwargs.pop('output_hidden_states', False)
        self.torchscript = kwargs.pop('torchscript', False)

    def save_pretrained(self, save_directory):
        """Save a configuration object to a directory, so that it can be re-
        loaded using the `from_pretrained(save_directory)` class method."""
        assert os.path.isdir(
            save_directory
        ), 'Saving path should be a directory where the model and ' \
           'configuration can be saved '

        # If we save using the predefined names, we can load using
        # `from_pretrained`
        output_config_file = os.path.join(save_directory, CONFIG_NAME)

        self.to_json_file(output_config_file)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        r""" Instantiate a PretrainedConfig from a pre-trained model
        configuration.

        Params: **pretrained_model_name_or_path**: either: - a string with
        the `shortcut name` of a pre-trained model configuration to load
        from cache or download and cache if not already stored in cache (
        e.g. 'bert-base-uncased'). - a path to a `directory` containing a
        configuration file saved using the `save_pretrained(save_directory)`
        method. - a path or url to a saved configuration `file`.
        **cache_dir**: (`optional`) string: Path to a directory in which a
        downloaded pre-trained model configuration should be cached if the
        standard cache should not be used. **return_unused_kwargs**: (
        `optional`) bool: - If False, then this function returns just the
        final configuration object. - If True, then this functions returns a
        tuple `(config, unused_kwargs)` where `unused_kwargs` is a
        dictionary consisting of the key/value pairs whose keys are not
        configuration attributes: ie the part of kwargs which has not been
        used to update `config` and is otherwise ignored. **kwargs**: (
        `optional`) dict: Dictionary of key/value pairs with which to update
        the configuration object after loading. - The values in kwargs of
        any keys which are configuration attributes will be used to override
        the loaded values. - Behavior concerning key/value pairs whose keys
        are *not* configuration attributes is controlled by the
        `return_unused_kwargs` keyword parameter.

        """
        cache_dir = kwargs.pop('cache_dir', None)
        return_unused_kwargs = kwargs.pop('return_unused_kwargs', False)

        if pretrained_model_name_or_path in cls.pretrained_config_archive_map:
            config_file = cls.pretrained_config_archive_map[
                pretrained_model_name_or_path]
        elif os.path.isdir(pretrained_model_name_or_path):
            config_file = os.path.join(pretrained_model_name_or_path,
                                       CONFIG_NAME)
        else:
            config_file = pretrained_model_name_or_path
        # redirect to the cache, if necessary
        try:
            resolved_config_file = cached_path(
                config_file, cache_dir=cache_dir)
        except EnvironmentError:
            if pretrained_model_name_or_path in \
                    cls.pretrained_config_archive_map:
                logger.error(
                    "Couldn't reach server at '{}' to download pretrained "
                    'model configuration file. '.format(config_file))
            else:
                logger.error(
                    "Model name '{}' was not found in model name list ({}). "
                    "We assumed '{}' was a path or url but couldn't find any "
                    'file '
                    'associated to this path or url.'.format(
                        pretrained_model_name_or_path,
                        ', '.join(cls.pretrained_config_archive_map.keys()),
                        config_file))
            return None
        if resolved_config_file == config_file:
            logger.info('loading configuration file {}'.format(config_file))
        else:
            logger.info(
                'loading configuration file {} from cache at {}'.format(
                    config_file, resolved_config_file))

        # Load config
        config = cls.from_json_file(resolved_config_file)

        # Update config with kwargs if needed
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                to_remove.append(key)
        # add img_layer_norm_eps, use_img_layernorm
        if 'img_layer_norm_eps' in kwargs:
            setattr(config, 'img_layer_norm_eps', kwargs['img_layer_norm_eps'])
            to_remove.append('img_layer_norm_eps')
        if 'use_img_layernorm' in kwargs:
            setattr(config, 'use_img_layernorm', kwargs['use_img_layernorm'])
            to_remove.append('use_img_layernorm')
        for key in to_remove:
            kwargs.pop(key, None)

        logger.info('Model config %s', config)
        if return_unused_kwargs:
            return config, kwargs
        else:
            return config

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `Config` from a Python dictionary of parameters."""
        config = cls(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, 'r', encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + '\n'

    def to_json_file(self, json_file_path):
        """Save this instance to a json file."""
        with open(json_file_path, 'w', encoding='utf-8') as writer:
            writer.write(self.to_json_string())


class BertConfig(PretrainedConfig):
    r"""
    :class:`~pytorch_transformers.BertConfig` is the configuration class to
    store the configuration of a `BertModel`.


        Arguments: vocab_size_or_config_json_file: Vocabulary size of
        `inputs_ids` in `BertModel`. hidden_size: Size of the encoder layers
        and the pooler layer. num_hidden_layers: Number of hidden layers in
        the Transformer encoder. num_attention_heads: Number of attention
        heads for each attention layer in the Transformer encoder.
        intermediate_size: The size of the "intermediate" (i.e.,
        feed-forward) layer in the Transformer encoder. hidden_act: The
        non-linear activation function (function or string) in the encoder
        and pooler. If string, "gelu", "relu" and "swish" are supported.
        hidden_dropout_prob: The dropout probabilitiy for all fully
        connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob: The dropout ratio for the attention
        probabilities. max_position_embeddings: The maximum sequence length
        that this model might ever be used with. Typically set this to
        something large just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size: The vocabulary size of the `token_type_ids` passed
        into `BertModel`. initializer_range: The sttdev of the
        truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps: The epsilon used by LayerNorm.
    """
    pretrained_config_archive_map = BERT_PRETRAINED_CONFIG_ARCHIVE_MAP

    def __init__(self,
                 vocab_size_or_config_json_file=30522,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act='gelu',
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02,
                 layer_norm_eps=1e-12,
                 **kwargs):
        super(BertConfig, self).__init__(**kwargs)
        if isinstance(vocab_size_or_config_json_file, str):
            with open(
                    vocab_size_or_config_json_file, 'r',
                    encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
            self.layer_norm_eps = layer_norm_eps
        else:
            raise ValueError(
                'First argument must be either a vocabulary size (int)'
                'or the path to a pretrained model config file (str)')


def _gelu_python(x):

    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
