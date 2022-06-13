# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import torch
from mmengine.model import BaseModel, merge_dict

from mmaction.core import ActionDataSample
from mmaction.registry import MODELS


ForwardResults = Union[Dict[str, torch.Tensor], List[ActionDataSample],
                       Tuple[torch.Tensor], torch.Tensor]


class BaseRecognizer(BaseModel, metaclass=ABCMeta):
    """Base class for recognizers.

    Args:
        backbone (dict): Backbone modules to extract feature.
        cls_head (dict | None): Classification head to process feature.
            Default: None.
        neck (dict | None): Neck for feature fusion. Default: None.
        train_cfg (dict | None): Config for training. Default: None.
        test_cfg (dict | None): Config for testing. Default: None.
        data_preprocessor (dict | None): Config for data preprocessor.
            Default: None.
    """

    def __init__(self,
                 backbone,
                 cls_head=None,
                 neck=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 data_preprocessor=None):
        if data_preprocessor is None:
            data_preprocessor = dict(type='ActionDataPreprocessor')

        super(BaseRecognizer, self).__init__(
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor)

        self.backbone = MODELS.build(backbone)

        if neck is not None:
            self.neck = MODELS.build(neck)

        if cls_head is not None:
            self.cls_head = MODELS.build(cls_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    @abstractmethod
    def extract_feat(self, batch_inputs, **kwargs):
        raise NotImplementedError

    @property
    def with_neck(self):
        """bool: whether the recognizer has a neck"""
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_cls_head(self):
        """bool: whether the recognizer has a cls_head"""
        return hasattr(self, 'cls_head') and self.cls_head is not None

    def init_weights(self):
        """Initialize the model network weights."""
        self.backbone.init_weights()
        if self.with_cls_head:
            self.cls_head.init_weights()
        if self.with_neck:
            self.neck.init_weights()

    def loss(self, batch_inputs, data_samples=None):
        feats, loss_kwargs = \
            self.extract_feat(batch_inputs, data_samples=data_samples)
        # loss_aux will be a empty dict if self.with_neck is False
        loss_aux = loss_kwargs.get('loss_aux', dict())
        loss_cls = self.cls_head.loss(feats, data_samples, **loss_kwargs)
        losses = merge_dict(loss_cls, loss_aux)
        return losses

    def predict(self, batch_inputs, data_samples=None):
        feats, predict_kwargs = self.extract_feat(batch_inputs, test_mode=True)
        predictions = self.cls_head.predict(feats, data_samples, **predict_kwargs)
        predictions = self.postprocess(predictions)
        return predictions

    def _forward(self, batch_inputs, stage='backbone') -> torch.Tensor:
        feats, _ = self.extract_feat(batch_inputs, stage=stage)
        return feats

    def forward(self,
                batch_inputs: torch.Tensor,
                data_samples: Optional[List[ActionDataSample]] = None,
                mode: str = 'feat') -> ForwardResults:
        if mode == 'feat':
            return self._forward(batch_inputs)
        if mode == 'predict':
            return self.predict(batch_inputs, data_samples)
        elif mode == 'loss':
            return self.loss(batch_inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}".')

    @staticmethod
    def postprocess(predictions) -> List[ActionDataSample]:
        """ Convert predictions to `ActionDataSample`. """
        for i in range(len(predictions)):
            result = ActionDataSample()
            result.pred_scores = predictions[i]
            predictions[i] = result
        return predictions
