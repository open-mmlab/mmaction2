# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List

from mmaction.core import ActionDataSample
from mmaction.registry import MODELS
from .base import BaseRecognizer


@MODELS.register_module()
class AudioRecognizer(BaseRecognizer):
    """Audio recognizer model framework."""

    def loss(self, inputs, data_samples) -> Dict:
        inputs = inputs.reshape((-1, ) + inputs.shape[2:])
        feats = self.extract_feat(inputs)
        assert self.with_cls_head, 'cls head must be implemented.'
        return self.cls_head.loss(feats, data_samples)

    def predict(self, inputs, data_samples) -> List[ActionDataSample]:
        inputs = inputs.reshape((-1, ) + inputs.shape[2:])
        x = self.extract_feat(audios)
        assert self.with_cls_head, 'cls head must be implemented.'
        return self.cls_head.predict(feats, data_samples)
