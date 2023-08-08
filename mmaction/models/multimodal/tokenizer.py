# Copyright (c) OpenMMLab. All rights reserved.
from transformers import BertTokenizer
from mmaction.registry import TOKENIZER

TOKENIZER.register_module('BertTokenizer', module=BertTokenizer.from_pretrained)
