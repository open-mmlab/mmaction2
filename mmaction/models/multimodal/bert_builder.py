# Copyright (c) OpenMMLab. All rights reserved.
from .xbert import BertConfig, BertForMaskedLM, BertLMHeadModel, BertModel

from mmaction.registry import MODELS

def build_bert(text_cfg, pretrain, checkpoint):
    """build text encoder.

    Args:
        model_config (dict): model config.
        pretrain (bool): Whether to do pretrain or finetuning.
        checkpoint (bool): whether to do gradient_checkpointing.

    Returns: TODO
    """
    bert_config = BertConfig.from_json_file(text_cfg.config)
    bert_config.encoder_width = text_cfg.d_model
    bert_config.gradient_checkpointing = checkpoint
    bert_config.fusion_layer = text_cfg.fusion_layer

    if not text_cfg.multimodal:
        bert_config.fusion_layer = bert_config.num_hidden_layers

    if pretrain:
        text_encoder, loading_info = BertForMaskedLM.from_pretrained(
            text_cfg.pretrained_model_name_or_path,
            config=bert_config,
            output_loading_info=True,
        )
    else:
        text_encoder, loading_info = BertModel.from_pretrained(
            text_cfg.pretrained_model_name_or_path,
            config=bert_config,
            add_pooling_layer=False,
            output_loading_info=True,
        )

    return text_encoder


def build_bert_decoder(text_cfg, checkpoint):
    """build text decoder the same as the multimodal encoder.

    Args:
        model_config (dict): model config.
        pretrain (bool): Whether to do pretrain or finetuning.
        checkpoint (bool): whether to do gradient_checkpointing.

    Returns: TODO
    """
    bert_config = BertConfig.from_json_file(text_cfg.config)
    bert_config.encoder_width = text_cfg.d_model
    bert_config.gradient_checkpointing = checkpoint

    bert_config.fusion_layer = 0
    bert_config.num_hidden_layers = (
        bert_config.num_hidden_layers - text_cfg.fusion_layer)

    text_decoder, loading_info = BertLMHeadModel.from_pretrained(
        text_cfg.pretrained_model_name_or_path,
        config=bert_config,
        output_loading_info=True,
    )

    return text_decoder

@MODELS.register_module()
class XBertForMaskedLM(BertForMaskedLM):

    def __init__(self, pretrained_model_name_or_path, fusion_layer, encoder_width, **kwargs):
        config = BertConfig.from_pretrained(pretrained_model_name_or_path)
        config.fusion_layer = fusion_layer
        config.encoder_width = encoder_width
        config.update(kwargs)
        super().__init__(config)

@MODELS.register_module()
class XBertModel(BertModel):

    def __init__(self, pretrained_model_name_or_path, fusion_layer, encoder_width, add_pooling_layer, **kwargs):
        config = BertConfig.from_pretrained(pretrained_model_name_or_path)
        config.fusion_layer = fusion_layer
        config.encoder_width = encoder_width
        config.update(kwargs)
        super().__init__(config, add_pooling_layer)



@MODELS.register_module()
class BertDecoder(BertLMHeadModel):

    def __init__(self, pretrained_model_name_or_path, fusion_layer, encoder_width, **kwargs):
        config = BertConfig.from_pretrained(pretrained_model_name_or_path)
        config.fusion_layer = fusion_layer
        config.encoder_width = encoder_width
        config.update(kwargs)
        super().__init__(config)

