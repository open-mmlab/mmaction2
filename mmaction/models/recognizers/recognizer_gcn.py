# Copyright (c) OpenMMLab. All rights reserved.
from torch import Tensor

from mmaction.registry import MODELS
from .base import BaseRecognizer


@MODELS.register_module()
class RecognizerGCN(BaseRecognizer):
    """GCN recognizer model framework."""

    def extract_feat(self,
                     inputs: Tensor,
                     stage: str = 'backbone',
                     **kwargs) -> tuple:
        """Extract features of different stages.

        Args:
            inputs (Tensor): The input data.
            stage (str): Which stage to output the feature.
                Defaults to ``backbone``.

        Returns:
            Tensor: The extracted features.
            dict: A dict recording the kwargs for downstream
                pipeline. This will be an empty dict in GCN recognizer.
        """

        # Record the kwargs required by `loss` and `predict`
        loss_predict_kwargs = dict()
        x = self.backbone(inputs)

        if stage == 'backbone':
            return x, loss_predict_kwargs

        if self.with_cls_head and stage == 'head':
            x = self.cls_head(x, **loss_predict_kwargs)
            return x, loss_predict_kwargs
