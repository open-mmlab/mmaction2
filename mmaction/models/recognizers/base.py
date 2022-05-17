# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Union

import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from mmcv.runner import auto_fp16
from mmaction.core import ActionDataSample, stack_batch
from ..builder import build_backbone, build_head, build_neck


class BaseRecognizer(BaseModule, metaclass=ABCMeta):
    """Base class for recognizers.

    Args:
        backbone (dict): Backbone modules to extract feature.
        cls_head (dict | None): Classification head to process feature.
            Default: None.
        neck (dict | None): Neck for feature fusion. Default: None.
        train_cfg (dict | None): Config for training. Default: None.
        test_cfg (dict | None): Config for testing. Default: None.
        init_cfg (dict, optional): Initialization config dict.
    """

    def __init__(self,
                 backbone,
                 cls_head=None,
                 neck=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        # record the source of the backbone
        self.backbone_from = 'mmaction2'

        if backbone['type'].startswith('mmcls.'):
            try:
                import mmcls.models.builder as mmcls_builder
            except (ImportError, ModuleNotFoundError):
                raise ImportError('Please install mmcls to use this backbone.')
            backbone['type'] = backbone['type'][6:]
            self.backbone = mmcls_builder.build_backbone(backbone)
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
            self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        if cls_head is not None:
            self.cls_head = build_head(cls_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # max_testing_views should be int
        self.max_testing_views = None
        if test_cfg is not None and 'max_testing_views' in test_cfg:
            self.max_testing_views = test_cfg['max_testing_views']
            assert isinstance(self.max_testing_views, int)

        if test_cfg is not None and 'feature_extraction' in test_cfg:
            self.feature_extraction = test_cfg['feature_extraction']
        else:
            self.feature_extraction = False

        # mini-batch blending, e.g. mixup, cutmix, etc.
        self.blending = None
        if train_cfg is not None and 'blending' in train_cfg:
            from mmcv.utils import build_from_cfg
            from mmaction.datasets.builder import BLENDINGS
            self.blending = build_from_cfg(train_cfg['blending'], BLENDINGS)

        self.fp16_enabled = False

    @property
    def device(self):
        return next(self.backbone.parameters()).device

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
        if self.backbone_from in ['mmcls', 'mmaction2']:
            self.backbone.init_weights()
        elif self.backbone_from in ['torchvision', 'timm']:
            warnings.warn('We do not initialize weights for backbones in '
                          f'{self.backbone_from}, since the weights for '
                          f'backbones in {self.backbone_from} are initialized'
                          'in their __init__ functions.')
        else:
            raise NotImplementedError('Unsupported backbone source '
                                      f'{self.backbone_from}!')

        if self.with_cls_head:
            self.cls_head.init_weights()
        if self.with_neck:
            self.neck.init_weights()

    @auto_fp16()
    def extract_feat(self, inputs):
        """Extract features through a backbone.

        Args:
            inputs (torch.Tensor): The input data.

        Returns:
            torch.tensor: The extracted features.
        """
        if (hasattr(self.backbone, 'features')
                and self.backbone_from == 'torchvision'):
            x = self.backbone.features(inputs)
        elif self.backbone_from == 'timm':
            x = self.backbone.forward_features(inputs)
        elif self.backbone_from == 'mmcls':
            x = self.backbone(inputs)
            if isinstance(x, tuple):
                assert len(x) == 1
                x = x[0]
        else:
            x = self.backbone(inputs)
        return x

    @abstractmethod
    def loss(self, inputs, data_samples) -> Dict:
        pass

    @abstractmethod
    def predict(self, inputs, data_samples) -> List[ActionDataSample]:
        pass

    def forward(self, data, return_loss=False) -> Union[Dict, List[ActionDataSample]]:
        """Define the computation performed at every call."""
        inputs, data_samples = self.preprocess_data(data)
        if return_loss:
            return self.loss(inputs, data_samples)
        else:
            return self.predict(inputs, data_samples)

    def preprocess_data(self, data):
        inputs = [data_['inputs'] for data_ in data]
        data_samples = [data_['data_sample'] for data_ in data]

        data_samples = [
            data_sample.to(self.device) for data_sample in data_samples
        ]
        inputs = [input.to(self.device) for input in inputs]
        batch_inputs = stack_batch(inputs)

        return batch_inputs, data_samples
