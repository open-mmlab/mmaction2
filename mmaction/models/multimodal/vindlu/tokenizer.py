# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

from transformers import BertTokenizer

from mmaction.registry import TOKENIZER


class VindLUTokenizer(BertTokenizer):
    """VindLUTokenizer inherit BertTokenizer.

    The main difference from BertTokenizer is removing the last separate token
    for a single sequence.
    """

    def build_inputs_with_special_tokens(
            self,
            token_ids_0: List[int],
            token_ids_1: Optional[List[int]] = None) -> List[int]:
        """Build model inputs from a sequence or a pair of sequence for
        sequence classification tasks by concatenating and adding special
        tokens. A BERT sequence has the following format:

        - single sequence: `[CLS] X`
        - pair of sequences: `[CLS] A [SEP] B [SEP]`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with
            the appropriate special tokens.
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep


TOKENIZER.register_module(
    'VindLUTokenizer', module=VindLUTokenizer.from_pretrained)
