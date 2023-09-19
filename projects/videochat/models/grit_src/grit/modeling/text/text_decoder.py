# Modified by Jialian Wu from
# https://github.com/microsoft/GenerativeImage2Text/blob/main/generativeimage2text/layers/decoder.py
# and https://github.com/kdexd/virtex
import functools
import warnings

import torch
from torch import nn
from torch.nn import functional as F


class TextualHead(nn.Module):

    def __init__(self, visual_feature_size: int, vocab_size: int,
                 hidden_size: int):
        super().__init__()
        self.visual_feature_size = visual_feature_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

    @property
    def textual_feature_size(self):
        return self.hidden_size


class WordAndPositionalEmbedding(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        dropout: float = 0.0,
        max_caption_length: int = 30,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx

        # self.words = nn.Embedding
        # (vocab_size, hidden_size, padding_idx=padding_idx)
        self.words = nn.Embedding(vocab_size, hidden_size)

        # We provide no "padding index" for positional embeddings. We zero out
        # the positional embeddings of padded positions as a post-processing.
        self.positions = nn.Embedding(max_caption_length, hidden_size)
        self.layer_norm = nn.LayerNorm(
            hidden_size, eps=1e-8, elementwise_affine=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, tokens: torch.Tensor):
        position_indices = self._create_position_indices(tokens)

        # shape: (batch_size, max_caption_length, hidden_size)
        word_embeddings = self.words(tokens)
        position_embeddings = self.positions(position_indices)

        # shape: (batch_size, max_caption_length, hidden_size)
        embeddings = self.layer_norm(word_embeddings + position_embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings

    @functools.lru_cache(maxsize=128)
    def _create_position_indices(self, tokens: torch.Tensor):

        # Create position indices of the same size as token indices.
        batch_size, max_caption_length = tokens.size()
        positions = torch.arange(
            max_caption_length, dtype=tokens.dtype, device=tokens.device)
        # shape: (batch_size, max_caption_length)
        positions = positions.unsqueeze(0).expand(batch_size,
                                                  max_caption_length)
        return positions


class BertEncoderAsDecoder(nn.Module):

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        tgt_bi_valid_mask=None,
        encoder_history_states=None,
    ):
        assert tgt_key_padding_mask is None, 'not supported'
        assert tgt_mask.dim() == 2
        assert tgt_mask.shape[0] == tgt_mask.shape[1]
        # tgt_mask should always be 0/negative infinity
        tgt = tgt.transpose(0, 1)
        memory = memory.transpose(0, 1)

        hidden_states = torch.cat((memory, tgt), dim=1)
        num_tgt = tgt.shape[1]
        num_memory = memory.shape[1]
        device = tgt.device
        dtype = tgt.dtype
        top_left = torch.zeros((num_memory, num_memory),
                               device=device,
                               dtype=dtype)
        top_right = torch.full(
            (num_memory, num_tgt),
            float('-inf'),
            device=tgt.device,
            dtype=dtype,
        )
        bottom_left = torch.zeros(
            (num_tgt, num_memory),
            dtype=dtype,
            device=tgt_mask.device,
        )
        left = torch.cat((top_left, bottom_left), dim=0)
        right = torch.cat((top_right, tgt_mask.to(dtype)), dim=0)

        full_attention_mask = torch.cat((left, right), dim=1)[None, :]

        if memory_key_padding_mask is None:
            memory_key_padding_mask = torch.full(
                (memory.shape[0], memory.shape[1]),
                fill_value=False,
                device=device)
        # if it is False, it means valid. That is, it is not a padding
        assert memory_key_padding_mask.dtype == torch.bool
        zero_negative_infinity = torch.zeros_like(
            memory_key_padding_mask, dtype=tgt.dtype)
        zero_negative_infinity[memory_key_padding_mask] = float('-inf')
        full_attention_mask = full_attention_mask.expand(
            (memory_key_padding_mask.shape[0], num_memory + num_tgt,
             num_memory + num_tgt))
        full_attention_mask = full_attention_mask.clone()
        origin_left = full_attention_mask[:, :, :num_memory]
        update = zero_negative_infinity[:, None, :]
        full_attention_mask[:, :, :num_memory] = origin_left + update

        if tgt_bi_valid_mask is not None:
            # verify the correctness
            bs = full_attention_mask.shape[0]
            # during inference, tgt_bi_valid_mask's length is not changed, but
            # num_tgt can be increased
            max_valid_target = tgt_bi_valid_mask.shape[1]
            mask = tgt_bi_valid_mask[:, None, :].expand(
                (bs, num_memory + num_tgt, max_valid_target))
            full_attention_mask[:, :, num_memory:(num_memory +
                                                  max_valid_target)][mask] = 0

        # add axis for multi-head
        full_attention_mask = full_attention_mask[:, None, :, :]

        if encoder_history_states is None:
            result = self.encoder(
                hidden_states=hidden_states,
                attention_mask=full_attention_mask,
                encoder_history_states=encoder_history_states,
            )
            result = list(result)
            result[0] = result[0][:, num_memory:].transpose(0, 1)
            if self.encoder.output_hidden_states:
                return result[0], result[1]
            else:
                # make it back-compatible
                return result[0]
        else:
            encoder_out = self.encoder(
                hidden_states=hidden_states[:, -1:],
                attention_mask=full_attention_mask[:, :, -1:],
                encoder_history_states=encoder_history_states,
            )
            result = encoder_out[0].transpose(0, 1)
            if self.encoder.output_hidden_states:
                return result, encoder_out[1]
            else:
                return result


def create_transformer(
    decoder_type,
    norm_type,
    textual_feature_size,
    attention_heads,
    feedforward_size,
    dropout,
    num_layers,
    output_hidden_states=False,
    use_mlp_wrapper=None,
    use_act_checkpoint=True,
):
    assert norm_type in ['post', 'pre']
    if decoder_type is None:
        LayerClass = (
            nn.TransformerDecoderLayer
            if norm_type == 'post' else PreNormTransformerDecoderLayer)
        _layer = LayerClass(
            textual_feature_size,
            attention_heads,
            dim_feedforward=feedforward_size,
            dropout=dropout,
            activation='gelu',
        )
        return nn.TransformerDecoder(_layer, num_layers)
    elif decoder_type == 'bert_en':
        from .modeling_bert import BertConfig, BertEncoder
        config = BertConfig(
            vocab_size_or_config_json_file=30522,
            hidden_size=textual_feature_size,
            num_hidden_layers=num_layers,
            num_attention_heads=attention_heads,
            intermediate_size=feedforward_size,
            hidden_act='gelu',
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            layer_norm_eps=1e-12,
        )
        config.pre_norm = (norm_type == 'pre')
        config.use_mlp_wrapper = use_mlp_wrapper
        config.output_hidden_states = output_hidden_states
        encoder = BertEncoder(config, use_act_checkpoint=use_act_checkpoint)
        return BertEncoderAsDecoder(encoder)


class PreNormTransformerDecoderLayer(nn.TransformerDecoderLayer):

    def forward(self,
                tgt,
                memory,
                tgt_mask=None,
                memory_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        # fmt: off
        # We use the members (modules) from super-class, just the order of
        # operations is changed here. First layernorm, then attention.
        tgt2 = self.norm1(tgt)
        tgt2, _ = self.self_attn(
            tgt2,
            tgt2,
            tgt2,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)

        # Layernorm first, then decoder attention.
        tgt2 = self.norm2(tgt)
        tgt2, _ = self.multihead_attn(
            tgt2,
            memory,
            memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)

        # Layernorm first, then transformation through feedforward network.
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


class TransformerDecoderTextualHead(TextualHead):

    def __init__(
        self,
        object_feature_size: int,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        attention_heads: int,
        feedforward_size: int,
        dropout: float = 0.1,
        norm_type: str = 'post',
        mask_future_positions: bool = True,
        max_caption_length: int = 1024,
        padding_idx: int = 0,
        decoder_type=None,
        not_tie_weight=None,
        output_hidden_states=None,
        use_mlp_wrapper=None,
        use_act_checkpoint=True,
    ):
        super().__init__(object_feature_size, vocab_size, hidden_size)
        self.num_layers = num_layers
        self.attention_heads = attention_heads
        self.feedforward_size = feedforward_size
        self.dropout = dropout
        assert mask_future_positions
        self.padding_idx = padding_idx

        self.object_feature_projection = nn.Sequential(
            nn.Linear(object_feature_size, self.textual_feature_size),
            nn.LayerNorm(self.textual_feature_size))

        self.embedding = WordAndPositionalEmbedding(
            self.vocab_size,
            self.textual_feature_size,
            dropout=dropout,
            max_caption_length=max_caption_length,
            padding_idx=padding_idx,
        )
        self.transformer = create_transformer(
            decoder_type=decoder_type,
            norm_type=norm_type,
            textual_feature_size=self.textual_feature_size,
            attention_heads=self.attention_heads,
            feedforward_size=self.feedforward_size,
            dropout=dropout,
            num_layers=self.num_layers,
            output_hidden_states=output_hidden_states,
            use_mlp_wrapper=use_mlp_wrapper,
            use_act_checkpoint=use_act_checkpoint,
        )
        self.apply(self._init_weights)

        # Create an output linear layer and tie the input and output word
        # embeddings to reduce parametejs.
        self.output = nn.Linear(self.textual_feature_size, vocab_size)
        if not not_tie_weight:
            self.output.weight = self.embedding.words.weight

    @staticmethod
    def _init_weights(module):
        """Initialize weights like BERT - N(0.0, 0.02), bias = 0."""

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.MultiheadAttention):
            module.in_proj_weight.data.normal_(mean=0.0, std=0.02)
            module.out_proj.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(
        self,
        hidden_states,
        text_tokens,
    ):
        projected_object_features = self.object_feature_projection(
            hidden_states) if hidden_states is not None else None
        batch_size, max_text_length = text_tokens.size()
        text_embeddings = self.embedding(text_tokens)

        # An additive mask for masking the future (one direction).
        uni_mask_zero_neg = self._generate_future_mask(max_text_length,
                                                       text_embeddings.dtype,
                                                       text_embeddings.device)

        # We transpose the first two dimensions of tokens embeddings and visual
        # features, as required by decoder.
        text_embeddings = text_embeddings.transpose(0, 1)

        projected_object_features = projected_object_features.transpose(0, 1)

        # if transformer here is the pytorch/decoder, there is no chance, the
        # output is always tensor
        trans_out = self.transformer(
            text_embeddings,
            projected_object_features,
            tgt_mask=uni_mask_zero_neg,
        )
        if isinstance(trans_out, tuple):
            textual_features = trans_out[0]
        else:
            assert isinstance(trans_out, torch.Tensor)
            textual_features = trans_out
        # Undo the transpose and bring batch to dim 0.
        # shape: (batch_size, max_caption_length, hidden_size)
        textual_features = textual_features.transpose(0, 1)

        # shape: (batch_size, max_caption_length, vocab_size)
        output_logits = self.output(textual_features)
        if isinstance(trans_out, tuple):
            return output_logits, trans_out[1]
        else:
            return output_logits

    def _generate_future_mask(self, size: int, dtype: torch.dtype,
                              device: torch.device):
        # Default mask is for forward direction. Flip for backward direction.
        mask = torch.triu(
            torch.ones(size, size, device=device, dtype=dtype), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask


class AutoRegressiveBeamSearch(object):

    def __init__(
        self,
        end_token_id: int,
        max_steps: int = 50,
        beam_size: int = 5,
        objectdet=True,
        per_node_beam_size: int = 2,
    ):
        self._eos_index = end_token_id
        self.max_steps = max_steps
        self.beam_size = beam_size
        self.objectdet = objectdet
        self.per_node_beam_size = per_node_beam_size or beam_size

    def search(self, begin_tokens, step):
        if self.beam_size > 1 and self.objectdet:
            only_return_best = False
        else:
            only_return_best = True

        batch_size = begin_tokens.size()[0]

        predictions = begin_tokens.unsqueeze(1).expand(
            (batch_size, self.beam_size, begin_tokens.shape[-1]))
        # Calculate the first timestep. This is done outside the main loop
        # because we are going from a single decoder input (the output from
        # the encoder) to the top `beam_size` decoder outputs. On the other
        # hand, within the main loop we are going from the `beam_size`
        # elements of the beam to `beam_size`^2 candidates from which we
        # will select the top `beam_size` elements for the next iteration.
        # shape: (batch_size, num_classes)
        start_class_logits = step(begin_tokens)

        # Convert logits to logprobs.
        # shape: (batch_size * beam_size, vocab_size)
        start_class_logprobs = F.log_softmax(start_class_logits, dim=1)

        num_classes = start_class_logprobs.size()[1]

        # shape: (batch_size, beam_size), (batch_size, beam_size)
        start_top_logprobs, start_predicted_classes = \
            start_class_logprobs.topk(self.beam_size)

        if (self.beam_size == 1
                and (start_predicted_classes == self._eos_index).all()):
            warnings.warn(
                'Empty object description predicted. You may want to '
                'increase beam '
                'size or ensure your step function is working properly.',
                RuntimeWarning,
            )
            if only_return_best:
                return start_predicted_classes, start_top_logprobs
            else:
                return start_predicted_classes.unsqueeze(
                    -1), start_top_logprobs

        # The log probs for the last time step.
        # shape: (batch_size, beam_size)
        last_logprobs = start_top_logprobs

        # shape: (batch_size, beam_size, sequence_length)
        predictions = torch.cat(
            [predictions, start_predicted_classes.unsqueeze(-1)], dim=-1)

        # Log probability tensor that mandates that the end token is selected.
        # shape: (batch_size * beam_size, num_classes)
        logprobs_after_end = start_class_logprobs.new_full(
            (batch_size * self.beam_size, num_classes), float('-inf'))
        logprobs_after_end[:, self._eos_index] = 0.0

        logits_after_end = start_class_logprobs.new_full(
            (batch_size * self.beam_size, num_classes), float('-inf'))
        logits_after_end[:, self._eos_index] = 0

        while predictions.shape[-1] < self.max_steps:
            # shape: (batch_size * beam_size,)
            last_predictions = predictions[:, :, -1].reshape(batch_size *
                                                             self.beam_size)

            # If every predicted token from the last step is `self._eos_index`,
            # then we can stop early.
            if (last_predictions == self._eos_index).all():
                break

            predictions_so_far = predictions.view(batch_size * self.beam_size,
                                                  -1)
            # shape: (batch_size * beam_size, num_classes)
            class_logits = step(predictions_so_far)

            # Set logprobs of last predicted tokens as high negative value
            # to avoid repetition in description.
            class_logits = class_logits.scatter(
                1, predictions_so_far[:, -1].view((-1, 1)), -10000)

            # shape: (batch_size * beam_size, num_classes)
            last_predictions_expanded = last_predictions.unsqueeze(-1).expand(
                batch_size * self.beam_size, num_classes)

            # Here we are finding any beams where we predicted the end token in
            # the previous timestep and replacing the distribution with a
            # one-hot distribution, forcing the beam to predict the end token
            # this timestep as well.
            class_logits = torch.where(
                last_predictions_expanded == self._eos_index,
                logits_after_end,
                class_logits,
            )

            # Convert logits to logprobs.
            # shape: (batch_size * beam_size, vocab_size)
            class_logprobs = F.log_softmax(class_logits, dim=1)

            # shape (both): (batch_size * beam_size, per_node_beam_size)
            top_logprobs, predicted_classes = class_logprobs.topk(
                self.per_node_beam_size)

            # Here we expand the last log probs to `(batch_size * beam_size,
            # per_node_beam_size)` so that we can add them to the current log
            # probs for this timestep. This lets us maintain the log
            # probability of each element on the beam.
            # shape: (batch_size * beam_size, per_node_beam_size)
            expanded_last_logprobs = (
                last_logprobs.unsqueeze(2).expand(
                    batch_size, self.beam_size,
                    self.per_node_beam_size).reshape(
                        batch_size * self.beam_size, self.per_node_beam_size))
            # shape: (batch_size * beam_size, per_node_beam_size)
            summed_top_logprobs = top_logprobs + expanded_last_logprobs

            # shape: (batch_size, beam_size * per_node_beam_size)
            reshaped_summed = summed_top_logprobs.reshape(
                batch_size, self.beam_size * self.per_node_beam_size)
            # shape: (batch_size, beam_size * per_node_beam_size)
            reshaped_predicted_classes = predicted_classes.reshape(
                batch_size, self.beam_size * self.per_node_beam_size)
            # Append the predictions to the current beam.
            reshaped_beam = (
                predictions.view(batch_size * self.beam_size, 1, -1).repeat(
                    1, self.per_node_beam_size,
                    1).reshape(batch_size,
                               self.beam_size * self.per_node_beam_size, -1))
            # batch_size, (beam_size * per_node_beach_size), #token
            reshaped_beam = torch.cat(
                [reshaped_beam,
                 reshaped_predicted_classes.unsqueeze(-1)],
                dim=-1)

            # Keep only the top `beam_size` beam indices.
            # shape: (batch_size, beam_size), (batch_size, beam_size)
            restricted_beam_logprobs, restricted_beam_indices = \
                reshaped_summed.topk(self.beam_size)
            predictions = reshaped_beam.gather(
                1,
                restricted_beam_indices.unsqueeze(-1).repeat(
                    1, 1, reshaped_beam.shape[-1]))

            # shape: (batch_size, beam_size)
            last_logprobs = restricted_beam_logprobs

        if not torch.isfinite(last_logprobs).all():
            warnings.warn(
                'Infinite log probs encountered. Some final descriptions may '
                'not '
                'make sense. This can happen when the beam size is larger than'
                ' the number of valid (non-zero probability) transitions that '
                'the step function produces.',
                RuntimeWarning,
            )

        # Optionally select best beam and its logprobs.
        if only_return_best:
            # shape: (batch_size, sequence_length)
            predictions = predictions[:, 0, :]
            last_logprobs = last_logprobs[:, 0]
        num_valid = (predictions != self._eos_index).sum(dim=-1)
        num_valid += (predictions == self._eos_index).sum(dim=-1) > 0
        num_valid = num_valid - begin_tokens.shape[1]
        num_valid = num_valid.clip(min=1)

        last_logprobs = last_logprobs / num_valid

        return predictions, last_logprobs


class GRiTTextDecoder(nn.Module):

    def __init__(
        self,
        transformer,
        begin_token_id=101,
        beamsearch_decode=None,
        loss_type=None,
        tokenizer=None,
    ):
        super().__init__()
        self.textual = transformer
        self.padding_idx = self.textual.padding_idx

        self.begin_token_id = begin_token_id
        self.beamsearch_decode = beamsearch_decode
        self.tokenizer = tokenizer

        if loss_type is None:
            self.loss = nn.CrossEntropyLoss(ignore_index=self.padding_idx)
        elif loss_type == 'smooth':
            self.loss = SmoothLabelCrossEntropyLoss(
                ignore_index=self.padding_idx)
        else:
            raise NotImplementedError(loss_type)

    def forward(self, batch):
        object_features = batch['object_features']

        if self.training:
            caption_token_input = batch['text_tokens']

            output_logits = self.textual(
                object_features,
                caption_token_input,
            )

            if 'need_predict' in batch:
                # in place should also be good, but we do not choose that for
                # safety as we may use it in prediction results in future
                target = batch['text_tokens'].clone()
                target[batch['need_predict'] == 0] = self.padding_idx
            else:
                target = batch['text_tokens']

            feat = output_logits[:, :-1].contiguous()
            target = target[:, 1:].contiguous()
            feat = feat.view(-1, self.textual.vocab_size)
            target = target.view(-1)

            valid_mask = target != self.padding_idx
            target = target[valid_mask]
            feat = feat[valid_mask]
            loss = self.loss(feat, target)

            return loss
        else:
            output_dict = self.infer(object_features)
        return output_dict

    def infer(self, object_features):
        batch_size = object_features.size(0)
        begin_tokens = object_features.new_full((batch_size, 1),
                                                self.begin_token_id).long()

        decoding_step = functools.partial(self.decoding_step, object_features)

        object_description_tokens, logprobs = self.beamsearch_decode.search(
            begin_tokens, decoding_step)

        output_dict = {
            'predictions': object_description_tokens,
            'logprobs': logprobs,
        }

        return output_dict

    def decoding_step(self, object_features, partial_text):
        batch_size = object_features.shape[0]
        beam_size = int(partial_text.size(0) / batch_size)
        if beam_size > 1:
            batch_size, num_token, channels = object_features.size()
            object_features = object_features.unsqueeze(1).repeat(
                1, beam_size, 1, 1)
            object_features = object_features.view(batch_size * beam_size,
                                                   num_token, channels)

        text_lengths = torch.ones_like(partial_text)
        if len(text_lengths.size()) != 2:
            partial_text = partial_text.unsqueeze(1)

        # shape: (batch_size * beam_size, partial_caption_length, vocab_size)
        logits = self.textual(
            object_features,
            partial_text,
        )

        return logits[:, -1, :].float()


class SmoothLabelCrossEntropyLoss(nn.Module):

    def __init__(self, eps=0.1, log_prefix='', ignore_index=None):
        super().__init__()
        self.eps = eps
        self.log_soft = nn.LogSoftmax(dim=1)
        self.kl = nn.KLDivLoss(reduction='none')

        self.iter = 0
        self.max_loss = 0
        self.min_loss = 0
        self.log_prefix = log_prefix
        self.ignore_index = ignore_index

    def forward(self, feature, target):
        feature = feature.float()
        if self.ignore_index is not None:
            valid_mask = target != self.ignore_index
            target = target[valid_mask]
            feature = feature[valid_mask]
        assert target.numel() > 0
        self.iter += 1
        eps = self.eps
        n_class = feature.size(1)
        one_hot = torch.zeros_like(feature).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = self.log_soft(feature)
        loss = self.kl(log_prb, one_hot)
        return loss.sum(dim=1).mean()
