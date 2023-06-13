# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from torch import Tensor

from mmaction.registry import MODELS
from mmaction.utils import SampleList
from .base import BaseRecognizer


@MODELS.register_module()
class Recognizer2D(BaseRecognizer):
    """2D recognizer model framework."""

    def extract_feat(self,
                     inputs: Tensor,
                     stage: str = 'neck',
                     data_samples: SampleList = None,
                     test_mode: bool = False) -> tuple:
        """Extract features of different stages.

        Args:
            inputs (Tensor): The input data.
            stage (str): Which stage to output the feature.
                Defaults to ``neck``.
            data_samples (List[:obj:`ActionDataSample`]): Action data
                samples, which are only needed in training. Defaults to None.
            test_mode: (bool): Whether in test mode. Defaults to False.

        Returns:
                Tensor: The extracted features.
                dict: A dict recording the kwargs for downstream
                    pipeline. These keys are usually included:
                    ``num_segs``, ``fcn_test``, ``loss_aux``.
        """

        # Record the kwargs required by `loss` and `predict`.
        loss_predict_kwargs = dict()

        num_segs = inputs.shape[1]
        loss_predict_kwargs['num_segs'] = num_segs

        # [N, num_crops * num_segs, C, H, W] ->
        # [N * num_crops * num_segs, C, H, W]
        # `num_crops` is calculated by:
        #   1) `twice_sample` in `SampleFrames`
        #   2) `num_sample_positions` in `DenseSampleFrames`
        #   3) `ThreeCrop/TenCrop` in `test_pipeline`
        #   4) `num_clips` in `SampleFrames` or its subclass if `clip_len != 1`
        inputs = inputs.view((-1, ) + inputs.shape[2:])

        # Check settings of `fcn_test`.
        fcn_test = False
        if test_mode:
            if self.test_cfg is not None and self.test_cfg.get(
                    'fcn_test', False):
                fcn_test = True
                num_segs = self.test_cfg.get('num_segs',
                                             self.backbone.num_segments)
            loss_predict_kwargs['fcn_test'] = fcn_test

        # Extract features through backbone.
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

        if self.backbone_from in ['torchvision', 'timm']:
            # Transformer-based feature shape: B x L x C.
            if len(x.shape) == 3 and x.shape[2] > 1:
                x = nn.AdaptiveAvgPool1d(1)(x.transpose(1, 2))  # B x C x 1
            # Resnet-based feature shape: B x C x Hs x Ws。
            if len(x.shape) == 4 and (x.shape[2] > 1 or x.shape[3] > 1):
                x = nn.AdaptiveAvgPool2d(1)(x)  # B x C x 1 x 1
            x = x.reshape((x.shape[0], -1))  # B x C
            x = x.reshape(x.shape + (1, 1))  # B x C x 1 x 1

        # Return features extracted through backbone.
        if stage == 'backbone':
            return x, loss_predict_kwargs

        loss_aux = dict()
        if self.with_neck:
            # x is a tuple with multiple feature maps.
            x = [
                each.reshape((-1, num_segs) +
                             each.shape[1:]).transpose(1, 2).contiguous()
                for each in x
            ]
            x, loss_aux = self.neck(x, data_samples=data_samples)
            if not fcn_test:
                x = x.squeeze(2)
                loss_predict_kwargs['num_segs'] = 1
        elif fcn_test:
            # full convolution (fcn) testing when no neck
            # [N * num_crops * num_segs, C', H', W'] ->
            # [N * num_crops, C', num_segs, H', W']
            x = x.reshape((-1, num_segs) +
                          x.shape[1:]).transpose(1, 2).contiguous()

        loss_predict_kwargs['loss_aux'] = loss_aux

        # Return features extracted through neck.
        if stage == 'neck':
            return x, loss_predict_kwargs

        # Return raw logits through head.
        if self.with_cls_head and stage == 'head':
            # [N * num_crops, num_classes]
            x = self.cls_head(x, **loss_predict_kwargs)
            return x, loss_predict_kwargs
