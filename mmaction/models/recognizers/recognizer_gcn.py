# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List

from mmaction.core import ActionDataSample
from ..builder import RECOGNIZERS
from .base import BaseRecognizer


@RECOGNIZERS.register_module()
class RecognizerGCN(BaseRecognizer):
    """GCN recognizer model framework."""

    def loss(self, inputs, data_samples) -> Dict:
        feats = self.extract_feat(inputs)
        assert self.with_cls_head, 'cls head must be implemented.'
        return self.cls_head.loss(feats, data_samples)

    def predict(self, inputs, data_samples) -> List[ActionDataSample]:
        feats = self.extract_feat(inputs)
        assert self.with_cls_head, 'cls head must be implemented.'
        return self.cls_head.predict(feats, data_samples)
