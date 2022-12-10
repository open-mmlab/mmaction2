# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch

from mmaction.registry import MODELS
from .base import BaseRecognizer


@MODELS.register_module()
class RecognizerGCN(BaseRecognizer):
    """GCN-based recognizer for skeleton-based action recognition."""

    def extract_feat(self,
                     inputs: torch.Tensor,
                     stage: str = 'backbone',
                     **kwargs) -> Tuple:
        """Extract features at the given stage.

        Args:
            inputs (torch.Tensor): The input skeleton with shape of
                `(B, num_clips, num_person, clip_len, num_joints, 3 or 2)`.
            stage (str): The stage to output the features.
                Defaults to ``'backbone'``.

        Returns:
            tuple: THe extracted features and a dict recording the kwargs
            for downstream pipeline, which is an empty dict for the
            GCN-based recognizer.
        """

        # Record the kwargs required by `loss` and `predict`
        loss_predict_kwargs = dict()

        bs, nc = inputs.shape[:2]
        inputs = inputs.reshape((bs * nc, ) + inputs.shape[2:])

        x = self.backbone(inputs)

        if stage == 'backbone':
            return x, loss_predict_kwargs

        if self.with_cls_head and stage == 'head':
            x = self.cls_head(x, **loss_predict_kwargs)
            return x, loss_predict_kwargs
