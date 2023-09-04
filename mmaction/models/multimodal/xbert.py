# Copyright (c) OpenMMLab. All rights reserved.
from mmaction.registry import MODELS
from .modeling_bert import (BertConfig, BertForMaskedLM, BertLMHeadModel,
                            BertModel)


@MODELS.register_module()
class XBertForMaskedLM(BertForMaskedLM):

    def __init__(self, pretrained_model_name_or_path, fusion_layer,
                 encoder_width, **kwargs):
        config = BertConfig.from_pretrained(pretrained_model_name_or_path)
        config.fusion_layer = fusion_layer
        config.encoder_width = encoder_width
        config.update(kwargs)
        super().__init__(config)


@MODELS.register_module()
class XBertModel(BertModel):

    def __init__(self, pretrained_model_name_or_path, fusion_layer,
                 encoder_width, add_pooling_layer, **kwargs):
        config = BertConfig.from_pretrained(pretrained_model_name_or_path)
        config.fusion_layer = fusion_layer
        config.encoder_width = encoder_width
        config.update(kwargs)
        super().__init__(config, add_pooling_layer)


@MODELS.register_module()
class BertDecoder(BertLMHeadModel):

    def __init__(self, pretrained_model_name_or_path, fusion_layer,
                 encoder_width, **kwargs):
        config = BertConfig.from_pretrained(pretrained_model_name_or_path)
        config.fusion_layer = fusion_layer
        config.encoder_width = encoder_width
        config.update(kwargs)
        super().__init__(config)
