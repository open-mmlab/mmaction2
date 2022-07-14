# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
import warnings
import torch
from torch import Tensor
from mmengine.model import BaseModel, merge_dict

from mmaction.core import ActionDataSample
from mmaction.registry import MODELS
ForwardResults = Union[Dict[str, torch.Tensor], List[ActionDataSample],
                       Tuple[torch.Tensor], torch.Tensor]
from mmaction.core.utils import (ConfigType, OptConfigType, OptMultiConfig,
                                 ForwardResults, SampleList, OptSampleList,
                                 LabelList)


class BaseRecognizer(BaseModel, metaclass=ABCMeta):
    """Base class for recognizers.

    Args:
        backbone (dict or ConfigDict): Backbone modules to extract feature.
        cls_head (dict or ConfigDict, optional): Classification head to
            process feature. Default: None.
        neck (dict or ConfigDict, optional): Neck for feature fusion.
            Default: None.
        train_cfg (dict or ConfigDict, optional): Config for training.
            Default: None.
        test_cfg (dict or ConfigDict, optional): Config for testing.
            Default: None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
           config of :class:`ActionDataPreprocessor`.  it usually includes,
            ``mean``, ``std`` and ``format_shape``.
        init_cfg (dict or ConfigDict, optional): Config to control the
           initialization. Default: None.
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
            warnings.warn(
                'The "data_preprocessor (dict) in BaseRecognizer is None. '
                'Please check whether it is defined in the config file. '
                'Here we adopt the default '
                '"data_preprocessor = dict(type=ActionDataPreprocessor)" '
                'to build. This may cause unexpected failure.')
            data_preprocessor = dict(type='ActionDataPreprocessor')

        super(BaseRecognizer, self).__init__(
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

        self.backbone = MODELS.build(backbone)

        if neck is not None:
            self.neck = MODELS.build(neck)

        if cls_head is not None:
            self.cls_head = MODELS.build(cls_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        assert 'average_clips' not in self.test_cfg, \
            'Average_clips (dict) is ' \
            'defined in the Head. Please see our document or the ' \
            'official config files.'
        

    @abstractmethod
    def extract_feat(self,
                     batch_inputs: Tensor,
                     **kwargs) -> ForwardResults:
        """Extract features from raw inputs."""
        raise NotImplementedError

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
        self.backbone.init_weights()
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
                as `gt_labels`.

        Returns:
            dict: A dictionary of loss components.
        """
        feats, loss_kwargs = \
            self.extract_feat(batch_inputs, data_samples=batch_data_samples)

        # loss_aux will be a empty dict if self.with_neck is False
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
            batch_data_samples (List[:obj:`ActionDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_labels`.

        Returns:
            List[:obj:`ActionDataSample`]: Return the recognition results.
            The returns value is ActionDataSample, which usually contain
            'pred_scores'. And the ``pred_scores`` usually contains
            following keys.

                - item (Tensor): Classification scores, has a shape
                    (num_classes, )
        """
        feats, predict_kwargs = self.extract_feat(batch_inputs, test_mode=True)
        predictions = self.cls_head.predict(feats, batch_data_samples,
                                            **predict_kwargs)
        # connvert to ActionDataSample
        predictions = self.convert_to_datasample(predictions)
        return predictions

    def _forward(self, batch_inputs: Tensor, stage: str = 'backbone',
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

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - "predict": Forward and return the predictions, which are fully
        processed to a list of :obj:`ActionDataSample`.
        - "loss": Forward and return a dict of losses according to the given
        inputs and data samples.

        Note that this method doesn't handle neither back propagation nor
        optimizer updating, which are done in the :meth:`train_step`.

        Args:
            batch_inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            batch_data_samples (List[:obj:`ActionDataSample`], optional): The
                annotation data of every samples. Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of :obj:`ActionDataSample`.
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
        """Convert predictions to `ActionDataSample`.

        Args:
            predictions (List[:obj:`LabelData`]): Recognition results wrapped
                by :obj:`LabelData`.

        Returns:
            List[:obj:`ActionDataSample`]: Recognition results wrapped by
            :obj:`ActionDataSample`. Each ActionDataSample usually contain
            'pred_scores'. And the ``pred_scores`` usually contains following keys.

                - item (Tensor): Classification scores, has a shape
                    (num_classes, )
        """
        predictions_list = []
        for i in range(len(predictions)):
            result = ActionDataSample()
            result.pred_scores = predictions[i]
            predictions_list.append(result)
        return predictions_list
