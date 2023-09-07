# Copyright (c) OpenMMLab. All rights reserved.
from abc import abstractmethod
from typing import Optional

import torch
from mmengine.logging import MMLogger
from mmengine.model import BaseModel
from mmengine.runner.checkpoint import _load_checkpoint
from torch import nn

from mmaction.registry import MODELS, TOKENIZER
from mmaction.utils import ForwardResults, SampleList
from .utils import (interpolate_pos_embed_beit,
                    interpolate_pos_relative_bias_beit)


class VindLUBase(BaseModel):
    """VindLU base Model.

    Args:
        tokenizer: (dict): The config for tokenizer.
        vision_encoder (dict): Backbone for extracting image features.
        text_encoder (dict): Backbone for extracting text features.
        temperature (float): Temperature parameter that controls the
            concentration level of the distribution. Defaults to 0.07.
        gradient_checkpointing (bool): Whether to do gradient_checkpointing.
            Using checkpoint will save some memory while slowing down the
            training speed. Defaults to False.
        data_preprocessor (Optional[dict]): The config for preprocessing input
            data.
        init_cfg (Optional[dict]): the config to control the initialization.
            Defaults to None.
    """

    def __init__(
        self,
        tokenizer: dict,
        vision_encoder: dict,
        text_encoder: dict,
        proj_dim: int = 256,
        temperature: float = 0.07,
        gradient_checkpointing: bool = False,
        pretrined_vl: bool = True,
        data_preprocessor: Optional[dict] = None,
        init_cfg: Optional[dict] = None,
    ):
        if data_preprocessor is None:
            data_preprocessor = dict(type='ActionDataPreprocessor')
        super().__init__(
            init_cfg=init_cfg, data_preprocessor=data_preprocessor)

        self.tokenizer = TOKENIZER.build(tokenizer)
        self.vision_cfg = vision_encoder
        self.text_encoder_cfg = text_encoder
        self.gradient_checkpointing = gradient_checkpointing
        self.text_encoder_cfg.gradient_checkpointing = gradient_checkpointing

        self.vision_width = vision_encoder.pop('encoder_width')
        self.text_width = text_encoder.encoder_width
        self.pretrined_vl = pretrined_vl

        if self.vision_cfg.pop('add_ln'):
            self.vision_layernorm = nn.LayerNorm(self.vision_width, eps=1e-12)
        else:
            self.vision_layernorm = nn.Identity()

        self.vision_encoder = MODELS.build(self.vision_cfg)

        if gradient_checkpointing:
            self.vision_encoder.gradient_checkpointing_enable()

        self.text_encoder = MODELS.build(self.text_encoder_cfg)

        self.vision_proj = nn.Linear(self.vision_width, proj_dim)
        self.text_proj = nn.Linear(self.text_width, proj_dim)

        self.temp = nn.parameter.Parameter(torch.ones([]) * temperature)
        self.itm_head = nn.Linear(self.text_width, 2)

    def extract_feat(self, inputs: torch.Tensor, **kwargs) -> ForwardResults:
        """Extract features from raw inputs."""

    @abstractmethod
    def loss(self, inputs: torch.Tensor, data_samples: SampleList,
             **kwargs) -> dict:
        """Calculate losses from a batch of inputs and data samples."""

    def forward(self, inputs, data_samples, mode: str = 'loss'):
        """The unified entry for a forward process in both training and test.

        The method should accept three modes:

        - ``tensor``: Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - ``predict``: Forward and return the predictions, which are fully
        processed to a list of :obj:`ActionDataSample`.
        - ``loss``: Forward and return a dict of losses according to the given
        inputs and data samples.

        Note that this method doesn't handle neither back propagation nor
        optimizer updating, which are done in the :meth:`train_step`.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (List[``ActionDataSample], optional): The
                annotation data of every samples. Defaults to None.
            mode (str): Return what kind of value. Defaults to ``tensor``.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of ``ActionDataSample``.
            - If ``mode="loss"``, return a dict of tensor.
        """

        if mode == 'tensor':
            return self.extract_feat(inputs, data_samples)
        elif mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}".')

    def encode_vision(self, image):
        """encode image / videos as features.

        Args:
            image (torch.Tensor): The input images.

        Returns: tuple.
            - vision_embeds (torch.Tensor): The features of all patches.
                Shape: [B,T,L,C].
            - pooled_vision_embeds (torch.Tensor): The pooled features.
                Shape: [B,T,C].
        """
        output_dict = self.vision_encoder(image)
        vision_embeds = self.vision_layernorm(output_dict.last_hidden_state)
        pooled_vision_embeds = output_dict.pooler_output

        return vision_embeds, pooled_vision_embeds

    def encode_text(self, text):
        """encode text.
        Args:
            text (dict): The output of huggingface's `PreTrainedTokenizer`.
                contains keys:
                - input_ids (torch.Tensor): Token ids to be fed to a model.
                    Shape: [B,L].
                - attention_mask (torch.Tensor): The mask indicate padded tokens.
                    Shape: [B,L]. 0 is padded token.
                - other keys refer to "https://huggingface.co/docs/transformers/v4.21.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer.__call__".  # noqa: E501
        Returns: tuple.
            - text_embeds (torch.Tensor): The features of all tokens. Shape: [B,L,C].
            - pooled_text_embeds (torch.Tensor): The pooled features. Shape: [B,C].

        """
        text_output = self.text_encoder(
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

    @property
    def device(self):
        return next(self.parameters()).device

    def preprocess_state_dict(self, state_dict):
        """Preprocess pretrained checkpoint for text_encoder."""
        for key in list(state_dict.keys()):
            if 'bert' in key:
                encoder_key = key.replace('bert.', '')
                state_dict[encoder_key] = state_dict[key]
                del state_dict[key]
        return state_dict

    def load_from_pretrainded_beit(self):
        from transformers.models.beit.modeling_beit import BeitModel
        beit2d = BeitModel.from_pretrained(
            self.vision_cfg.pretrained_model_name_or_path)
        ori_state_dict = beit2d.state_dict()
        del beit2d
        # interpolate relative pos bias
        state_dict = interpolate_pos_relative_bias_beit(
            state_dict_old=ori_state_dict,
            state_dict_new=self.vision_encoder.state_dict(),
            patch_shape_new=self.vision_encoder.embeddings.patch_embeddings.
            patch_shape,
        )

        for k in list(state_dict.keys()):
            if 'prompt_bias_table' in k:
                del state_dict[k]

        msg = self.vision_encoder.load_state_dict(state_dict, strict=False)
        logger = MMLogger.get_current_instance()
        logger.info(msg)

    def init_weights(self):
        if self.vision_cfg.get('pretrained2d', False):
            self.load_from_pretrainded_beit()

        if self.pretrined_vl:
            assert self.init_cfg.get('type') == 'Pretrained', (
                'Please specify '
                'init_cfg to use pretrained video-language checkpoint')
            self.pretrained = self.init_cfg.get('checkpoint')
            checkpoint = _load_checkpoint(self.pretrained, map_location='cpu')
            state_dict = checkpoint['model']
            state_dict = interpolate_pos_embed_beit(state_dict, self)
            state_dict = self.preprocess_state_dict(state_dict)
            msg = self.load_state_dict(state_dict, strict=False)
            logger = MMLogger.get_current_instance()
            logger.info(msg)
        else:
            super().init_weights()
