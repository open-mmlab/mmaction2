# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from abc import ABCMeta, abstractmethod

import torch.nn as nn
from mmengine.model import BaseModel, merge_dict
from torch import Tensor

from mmaction.data_elements import ActionDataSample
from mmaction.registry import MODELS
from mmaction.utils import (ConfigType, ForwardResults, LabelList,
                            OptConfigType, OptMultiConfig, OptSampleList,
                            SampleList)


class BaseRecognizer(BaseModel, metaclass=ABCMeta):
    """Base class for recognizers.

    Args:
        backbone (dict or ConfigDict): Backbone modules to extract feature.
        cls_head (dict or ConfigDict, optional): Classification head to
            process feature. Defaults to None.
        neck (dict or ConfigDict, optional): Neck for feature fusion.
            Defaults to None.
        train_cfg (dict or ConfigDict, optional): Config for training.
            Defaults to None.
        test_cfg (dict or ConfigDict, optional): Config for testing.
            Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
           config of :class:`ActionDataPreprocessor`.  it usually includes,
            ``mean``, ``std`` and ``format_shape``. Defaults to None.
        init_cfg (dict or ConfigDict, optional): Config to control the
           initialization. Defaults to None.
    """

    def __init__(self,
                 backbone: ConfigType,
                 cls_head: OptConfigType = None,
                 neck: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        if data_preprocessor is None:
            # This preprocessor will only stack batch data samples.
            data_preprocessor = dict(type='ActionDataPreprocessor')

        super(BaseRecognizer, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        # record the source of the backbone
        self.backbone_from = 'mmaction2'

        if backbone['type'].startswith('mmcls.'):
            try:
                # Register all mmcls models.
                import mmcls.models  # noqa: F401
            except (ImportError, ModuleNotFoundError):
                raise ImportError('Please install mmcls to use this backbone.')
            self.backbone = MODELS.build(backbone)
            self.backbone_from = 'mmcls'
        elif backbone['type'].startswith('torchvision.'):
            try:
                import torchvision.models
            except (ImportError, ModuleNotFoundError):
                raise ImportError('Please install torchvision to use this '
                                  'backbone.')
            backbone_type = backbone.pop('type')[12:]
            self.backbone = torchvision.models.__dict__[backbone_type](
                **backbone)
            # disable the classifier
            self.backbone.classifier = nn.Identity()
            self.backbone.fc = nn.Identity()
            self.backbone_from = 'torchvision'
        elif backbone['type'].startswith('timm.'):
            try:
                import timm
            except (ImportError, ModuleNotFoundError):
                raise ImportError('Please install timm to use this '
                                  'backbone.')
            backbone_type = backbone.pop('type')[5:]
            # disable the classifier
            backbone['num_classes'] = 0
            self.backbone = timm.create_model(backbone_type, **backbone)
            self.backbone_from = 'timm'
        else:
            self.backbone = MODELS.build(backbone)

        if neck is not None:
            self.neck = MODELS.build(neck)

        if cls_head is not None:
            self.cls_head = MODELS.build(cls_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    @abstractmethod
    def extract_feat(self, batch_inputs: Tensor, **kwargs) -> ForwardResults:
        """Extract features from raw inputs."""

    @property
    def with_neck(self) -> bool:
        """bool: whether the recognizer has a neck"""
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_cls_head(self) -> bool:
        """bool: whether the recognizer has a cls_head"""
        return hasattr(self, 'cls_head') and self.cls_head is not None

    def init_weights(self) -> None:
        """Initialize the model network weights."""
        if self.backbone_from in ['mmcls', 'mmaction2']:
            self.backbone.init_weights()
        elif self.backbone_from in ['torchvision', 'timm']:
            warnings.warn('We do not initialize weights for backbones in '
                          f'{self.backbone_from}, since the weights for '
                          f'backbones in {self.backbone_from} are initialized'
                          'in their __init__ functions.')

        if self.with_cls_head:
            self.cls_head.init_weights()
        if self.with_neck:
            self.neck.init_weights()

    def loss(self, batch_inputs: Tensor, batch_data_samples: SampleList,
             **kwargs) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Raw Inputs of the recognizer.
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`ActionDataSample`]): The batch
                data samples. It usually includes information such
                as ``gt_labels``.

        Returns:
            dict: A dictionary of loss components.
        """
        feats, loss_kwargs = \
            self.extract_feat(batch_inputs,
                              batch_data_samples=batch_data_samples)

        # loss_aux will be a empty dict if `self.with_neck` is False.
        loss_aux = loss_kwargs.get('loss_aux', dict())
        loss_cls = self.cls_head.loss(feats, batch_data_samples, **loss_kwargs)
        losses = merge_dict(loss_cls, loss_aux)
        return losses

    def predict(self, batch_inputs: Tensor, batch_data_samples: SampleList,
                **kwargs) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Raw Inputs of the recognizer.
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`ActionDataSample`]): The batch
                data samples. It usually includes information such
                as ``gt_labels``.

        Returns:
            List[:obj:`ActionDataSample`]: Return the recognition results.
            The returns value is ``ActionDataSample``, which usually contains
            ``pred_scores``. And the ``pred_scores`` usually contains
            following keys.

                - item (Tensor): Classification scores, has a shape
                    (num_classes, )
        """
        feats, predict_kwargs = self.extract_feat(batch_inputs, test_mode=True)
        predictions = self.cls_head.predict(feats, batch_data_samples,
                                            **predict_kwargs)
        # convert to ActionDataSample.
        predictions = self.convert_to_datasample(predictions)
        return predictions

    def _forward(self,
                 batch_inputs: Tensor,
                 stage: str = 'backbone',
                 **kwargs) -> ForwardResults:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            batch_inputs (Tensor): Raw Inputs of the recognizer.
            stage (str): Which stage to output the features.

        Returns:
            tuple or Tensor: Features from ``backbone`` or ``neck`` or ``head``
            forward.
        """
        feats, _ = self.extract_feat(batch_inputs, stage=stage)
        return feats

    def forward(self,
                batch_inputs: Tensor,
                batch_data_samples: OptSampleList = None,
                mode: str = 'tensor',
                **kwargs) -> ForwardResults:
        """The unified entry for a forward process in both training and test.

        The method should accept three modes:

        - ``tensor``: Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - ``predict``: Forward and return the predictions, which are fully
        processed to a list of :obj:`ActionDataSample`.
        - ``loss``: Forward and return a dict of losses according to the given
        inputs and data samples.

        Note that this method doesn't handle neither back propagation nor
        optimizer updating, which are done in the :meth:`train_step`.

        Args:
            batch_inputs (Tensor): The input tensor with shape
                (N, C, ...) in general.
            batch_data_samples (List[:obj:`ActionDataSample`], optional): The
                annotation data of every samples. Defaults to None.
            mode (str): Return what kind of value. Defaults to ``tensor``.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of ``ActionDataSample``.
            - If ``mode="loss"``, return a dict of tensor.
        """
        if mode == 'tensor':
            return self._forward(batch_inputs, **kwargs)
        if mode == 'predict':
            return self.predict(batch_inputs, batch_data_samples, **kwargs)
        elif mode == 'loss':
            return self.loss(batch_inputs, batch_data_samples, **kwargs)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')

    def convert_to_datasample(self, predictions: LabelList) -> SampleList:
        """Convert predictions to ``ActionDataSample``.

        Args:
            predictions (List[:obj:`LabelData`]): Recognition results wrapped
                by ``LabelData``.

        Returns:
            List[:obj:`ActionDataSample`]: Recognition results wrapped by
            ``ActionDataSample``.
        """
        predictions_list = []
        for i in range(len(predictions)):
            result = ActionDataSample()
            result.pred_scores = predictions[i]
            predictions_list.append(result)
        return predictions_list
