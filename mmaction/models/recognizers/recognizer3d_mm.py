# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Tuple

import torch

from mmaction.registry import MODELS
from mmaction.utils import OptSampleList
from .base import BaseRecognizer


@MODELS.register_module()
class MMRecognizer3D(BaseRecognizer):
    """Multi-modal 3D recognizer model framework."""

    def extract_feat(self,
                     inputs: Dict[str, torch.Tensor],
                     stage: str = 'backbone',
                     data_samples: OptSampleList = None,
                     test_mode: bool = False) -> Tuple:
        """Extract features.

        Args:
            inputs (dict[str, torch.Tensor]): The multi-modal input data.
            stage (str): Which stage to output the feature.
                Defaults to ``'backbone'``.
            data_samples (list[:obj:`ActionDataSample`], optional): Action data
                samples, which are only needed in training. Defaults to None.
            test_mode (bool): Whether in test mode. Defaults to False.

        Returns:
                tuple[torch.Tensor]: The extracted features.
                dict: A dict recording the kwargs for downstream
                    pipeline.
        """
        # [N, num_views, C, T, H, W] ->
        # [N * num_views, C, T, H, W]
        for m, m_data in inputs.items():
            m_data = m_data.reshape((-1, ) + m_data.shape[2:])
            inputs[m] = m_data

        # Record the kwargs required by `loss` and `predict`
        loss_predict_kwargs = dict()

        x = self.backbone(**inputs)
        if stage == 'backbone':
            return x, loss_predict_kwargs

        if self.with_cls_head and stage == 'head':
            x = self.cls_head(x, **loss_predict_kwargs)
            return x, loss_predict_kwargs
