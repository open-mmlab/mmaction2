# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List

import torch
from torch import nn

from mmaction.core import ActionDataSample
from ..builder import RECOGNIZERS
from .base import BaseRecognizer


@RECOGNIZERS.register_module()
class Recognizer2D(BaseRecognizer):
    """2D recognizer model framework."""

    def loss(self, inputs, data_samples) -> Dict:
        assert self.with_cls_head, 'cls head must be implemented.'

        batches = inputs.shape[0]
        inputs = inputs.reshape((-1, ) + inputs.shape[2:])
        num_segs = inputs.shape[0] // batches

        feats = self.extract_feat(inputs)

        if self.backbone_from in ['torchvision', 'timm']:
            if len(feats.shape) == 4 and (feats.shape[2] > 1 or feats.shape[3] > 1):
                # apply adaptive avg pooling
                feats = nn.AdaptiveAvgPool2d(1)(feats)
            feats = feats.reshape((feats.shape[0], -1))
            feats = feats.reshape(feats.shape + (1, 1))

        loss_aux = None
        if self.with_neck:
            feats = [
                each.reshape((-1, num_segs) +
                             each.shape[1:]).transpose(1, 2).contiguous()
                for each in feats
            ]
            feats, loss_aux = self.neck(feats, data_samples)
            feats = feats.squeeze(2)
            num_segs = 1
            losses.update(loss_aux)

        return self.cls_head.loss(feats, data_samples,
                                  loss_aux=loss_aux, num_segs=num_segs)

    def predict(self, inputs, data_samples) -> List[ActionDataSample]:
        batches = inputs.shape[0]
        inputs = inputs.reshape((-1, ) + inputs.shape[2:])
        num_segs = inputs.shape[0] // batches

        feats = self.extract_feat(inputs)

        if self.backbone_from in ['torchvision', 'timm']:
            if len(feats.shape) == 4 and (feats.shape[2] > 1 or feats.shape[3] > 1):
                # apply adaptive avg pooling
                feats = nn.AdaptiveAvgPool2d(1)(feats)
            feats = feats.reshape((feats.shape[0], -1))
            feats = feats.reshape(feats.shape + (1, 1))

        if self.with_neck:
            feats = [
                each.reshape((-1, num_segs) +
                             each.shape[1:]).transpose(1, 2).contiguous()
                for each in feats
            ]
            feats, _ = self.neck(feats)
            feats = feats.squeeze(2)
            num_segs = 1

        if self.feature_extraction:
            # perform spatial pooling
            avg_pool = nn.AdaptiveAvgPool2d(1)
            feats = avg_pool(feats)
            # squeeze dimensions
            feats = feats.reshape((batches, num_segs, -1))
            # temporal average pooling
            feats = feats.mean(axis=1)
            return feats

        # When using `TSNHead` or `TPNHead`, shape is [batch_size, num_classes]
        # When using `TSMHead`, shape is [batch_size * num_crops, num_classes]
        # `num_crops` is calculated by:
        #   1) `twice_sample` in `SampleFrames`
        #   2) `num_sample_positions` in `DenseSampleFrames`
        #   3) `ThreeCrop/TenCrop` in `test_pipeline`
        #   4) `num_clips` in `SampleFrames` or its subclass if `clip_len != 1`

        # should have cls_head if not extracting features
        assert self.with_cls_head, 'cls head must be implemented.'
        return self.cls_head.predict(feats, data_samples, num_segs=num_segs)
