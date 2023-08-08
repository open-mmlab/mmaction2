# Copyright (c) OpenMMLab. All rights reserved.
import imp
import logging

import mmengine
import torch
from einops import rearrange
from mmengine.logging import MMLogger
from mmengine.model import BaseModel
from mmengine.runner.checkpoint import _load_checkpoint
from torch import nn
from zmq import device

from .beit_builder import build_beit, build_beit3d, interpolate_pos_embed_beit
from .bert_builder import build_bert
from .tokenization_bert import BertTokenizer

from mmaction.registry import TOKENIZER, MODELS
# from .criterions import MLMLoss, VTC_VTM_Loss


class VindLU(BaseModel):
    """docstring for VindLU."""

    def __init__(self, 
                 tokenizer, 
                 vision_encoder, 
                 text_encoder,
                 text_decoder,
                 proj_dim,
                 temperature, 
                 evaluate, 
                 gradient_checkpointing,
                 answer_list_path, 
                 pretrained_ckpt=None,
                 data_preprocessor=None,):
        if data_preprocessor is None:
            # This preprocessor will only stack batch data samples.
            data_preprocessor = dict(type='ActionDataPreprocessor')
        super(VindLU, self).__init__(data_preprocessor=data_preprocessor)

        # self.config = config
        # self.tokenizer = BertTokenizer.from_pretrained(
        #     tokenizer.pretrained_model_name_or_path)
        # cant align with vindlu's BertTokenizer, hf's BertTokenizer will add a sep token
        self.tokenizer = TOKENIZER.build(tokenizer)
        self.vision_cfg = vision_encoder
        self.text_encoder_cfg = text_encoder
        self.text_decoder_cfg = text_decoder
        self.evaluate = evaluate
        self.gradient_checkpointing = gradient_checkpointing
        self.vision_width = vision_encoder.encoder_width
        self.text_width = text_encoder.encoder_width
        self.pretrained_ckpt = pretrained_ckpt

        if answer_list_path:
            self.answer_list = mmengine.load(answer_list_path)

        # create modules.
        self.vision_encoder, self.vision_layernorm = self.build_vision_encoder(
        )

        # self.vision_encoder = build_beit3d(self.vision_cfg, self.gradient_checkpointing)
        # self.text_encoder = self.build_text_encoder()
        self.text_encoder = MODELS.build(self.text_encoder_cfg)

        self.vision_proj = nn.Linear(self.vision_width, proj_dim)
        self.text_proj = nn.Linear(self.text_width, proj_dim)

        self.temp = nn.parameter.Parameter(torch.ones([]) * temperature)
        self.itm_head = nn.Linear(self.text_width, 2)

        # criterions
        # self.loss_weight = config.criterion.loss_weight
        self.loss_weight = None
        # self.criterion_vtc_vtm = VTC_VTM_Loss(config.criterion.vtm_hard_neg)
        # self.criterion_mlm = MLMLoss(config.criterion.mlm_masking_prob, tokenizer)

    def forward(self, inputs, data_samples, mode: str = 'loss'):
        """forward and calculate loss.

        Args:
            image (torch.Tensor): The input images. Shape: [B,T,C,H,W].
            text (dict): TODO
            idx (torch.Tensor): TODO

        Returns: TODO
        """
        self.clip_contrastive_temperature()

        vision_embeds, pooled_vision_embeds = self.vision_encoder(inputs)
        text_embeds, pooled_text_embeds = self.encode_text(text)

        # obtain vision and text representations.
        vision_proj = self.vision_proj(pooled_vision_embeds)
        text_proj = self.text_proj(pooled_text_embeds)

        # calculate loss

        ## VTC loss
        # if self.loss_weight.vtc != 0:
        #     loss_vtc = self.criterion_vtc_vtm.vtc_loss(
        #         vision_proj, text_proj, idx, self.temp, all_gather=True
        #     )
        # else:
        loss_vtc = torch.tensor(0)

        vision_embeds = rearrange(vision_embeds, 'b t l c -> b (t l) c')

        ## VTM loss
        # if self.loss_weight.vtm != 0:
        #     loss_vtm = self.criterion_vtc_vtm.vtm_loss(
        #         self.get_text_encoder(),
        #         self.itm_head,
        #         self.temp,
        #         vision_embeds,
        #         text_embeds,
        #         vision_proj,
        #         text_proj,
        #         text.attention_mask,
        #         idx,
        #     )
        # else:
        loss_vtm = torch.tensor(0)

        ## MLM loss
        if self.text_encoder_cfg.is_pretrain and self.loss_weight.mlm != 0:
            loss_mlm = self.criterion_mlm.mlm_loss(self.text_encoder, text,
                                                   vision_embeds, None)
        else:
            loss_mlm = torch.tensor(0)

        return dict(
            loss_vtc=loss_vtc * self.loss_weight.vtc,
            loss_vtm=loss_vtm * self.loss_weight.vtm,
            loss_mlm=loss_mlm * self.loss_weight.mlm,
        )

    def encode_vision(self, image):
        """encode image / videos as features.

        Args:
            image (torch.Tensor): The input images.

        Returns: tuple.
            - vision_embeds (torch.Tensor): The features of all patches. Shape: [B,T,L,C].
            - pooled_vision_embeds (torch.Tensor): The pooled features. Shape: [B,T,C].
        """
        output_dict = self.vision_encoder(image)
        vision_embeds = self.vision_layernorm(output_dict.last_hidden_state)
        pooled_vision_embeds = output_dict.pooler_output

        return vision_embeds, pooled_vision_embeds

    def encode_text(self, text):
        """encode text.
        Args:
            text (dict): The output of huggingface's `PreTrainedTokenizer`. contains keys:
                - input_ids (torch.Tensor): Token ids to be fed to a model. Shape: [B,L].
                - attention_mask (torch.Tensor): The mask indicate padded tokens. Shape: [B,L]. 0 is padded token.
                - other keys refer to "https://huggingface.co/docs/transformers/v4.21.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer.__call__".
        Returns: tuple.
            - text_embeds (torch.Tensor): The features of all tokens. Shape: [B,L,C].
            - pooled_text_embeds (torch.Tensor): The pooled features. Shape: [B,C].

        """
        text_output = self.get_text_encoder()(
            text.input_ids,
            attention_mask=text.attention_mask,
            return_dict=True,
            mode='text',
        )
        text_embeds = text_output.last_hidden_state
        pooled_text_embeds = text_embeds[:, 0]
        return text_embeds, pooled_text_embeds

    @torch.no_grad()
    def clip_contrastive_temperature(self, min_val=0.001, max_val=0.5):
        """Seems only used during pre-training."""
        self.temp.clamp_(min_val, max_val)

    def build_vision_encoder(self):
        """build vision encoder
        Returns: (vision_encoder, vision_layernorm). Each is a `nn.Module`.

        """
        encoder_name = self.vision_cfg.type
        logger = MMLogger.get_current_instance()
        logger.info(f'Build vision_encoder: {encoder_name}')
        if 'beit' in encoder_name:
            vision_encoder = build_beit3d(
                self.vision_cfg,
                self.gradient_checkpointing,
            )
            # vision_encoder = build_beit(
            #     self.vision_cfg,
            #     224,
            #     self.gradient_checkpointing,
            # )

        else:
            raise ValueError(f'not implemented: {encoder_name}')

        if self.vision_cfg.add_ln:
            vision_layernorm = nn.LayerNorm(self.vision_width, eps=1e-12)
        else:
            vision_layernorm = nn.Identity()
        return vision_encoder, vision_layernorm

    def build_text_encoder(self):
        """build text_encoder and possibly video-to-text multimodal fusion
        encoder.

        Returns: nn.Module. The text encoder
        """
        encoder_name = self.text_encoder_cfg.type
        logger = MMLogger.get_current_instance()
        logger.info(f'Build text_encoder {encoder_name}')

        if 'bert' in encoder_name:
            text_encoder = build_bert(
                self.text_encoder_cfg,
                self.text_encoder_cfg.is_pretrain,
                self.gradient_checkpointing,
            )
        else:
            raise ValueError(f'Not implemented: {encoder_name}')

        return text_encoder

    def get_text_encoder(self):
        """get text encoder, used for text and cross-modal encoding."""
        encoder = self.text_encoder
        return encoder.bert if hasattr(encoder, 'bert') else encoder

    def init_weights(self):
        checkpoint = _load_checkpoint(self.pretrained_ckpt, map_location='cpu')
        state_dict = checkpoint['model']
        state_dict = interpolate_pos_embed_beit(state_dict, self)
        for key in list(state_dict.keys()):
            if 'bert' in key:
                encoder_key = key.replace('bert.', '')
                state_dict[encoder_key] = state_dict[key]
                if not self.text_decoder_cfg:
                    del state_dict[key]

            # init text decoder as multimodal encoder (last 6 layers of model.text_encoder)
            # only for generation tasks like VQA
            if self.text_decoder_cfg and 'text_encoder' in key:
                if 'layer' in key:
                    encoder_keys = key.split('.')
                    layer_num = int(encoder_keys[4])
                    if layer_num < self.text_encoder_cfg.fusion_layer:
                        del state_dict[key]
                        continue
                    else:
                        decoder_layer_num = layer_num - 9
                        encoder_keys[4] = str(decoder_layer_num)
                        encoder_key = '.'.join(encoder_keys)
                else:
                    encoder_key = key
                decoder_key = encoder_key.replace('text_encoder',
                                                    'text_decoder')
                state_dict[decoder_key] = state_dict[key]
                del state_dict[key]
        msg = self.load_state_dict(state_dict, strict=False)
        logger = MMLogger.get_current_instance()
        logger.info(msg)
