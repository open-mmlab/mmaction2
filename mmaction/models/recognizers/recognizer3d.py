# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List

import torch
from torch import nn

from mmaction.core import ActionDataSample
from ..builder import RECOGNIZERS
from .base import BaseRecognizer


@RECOGNIZERS.register_module()
class Recognizer3D(BaseRecognizer):
    """3D recognizer model framework."""

    def loss(self, inputs, data_samples) -> Dict:
        assert self.with_cls_head, 'cls head must be implemented.'

        inputs = inputs.reshape((-1, ) + inputs.shape[2:])
        feats = self.extract_feat(inputs)

        # TODO
        loss_aux = None
        if self.with_neck:
            feats, loss_aux = self.neck(feats, data_samples)

        return self.cls_head.loss(feats, data_samples, loss_aux=loss_aux)

    def predict(self, inputs, data_samples) -> List[ActionDataSample]:
        batches = inputs.shape[0]
        num_segs = inputs.shape[1]
        inputs = inputs.reshape((-1, ) + inputs.shape[2:])

        if self.max_testing_views is not None:
            total_views = inputs.shape[0]
            assert num_segs == total_views, (
                'max_testing_views is only compatible '
                'with batch_size == 1')
            view_ptr = 0
            feats = []
            while view_ptr < total_views:
                batch_imgs = inputs[view_ptr:view_ptr + self.max_testing_views]
                x = self.extract_feat(batch_imgs)
                if self.with_neck:
                    x, _ = self.neck(x)
                feats.append(x)
                view_ptr += self.max_testing_views
            # should consider the case that feat is a tuple
            if isinstance(feats[0], tuple):
                len_tuple = len(feats[0])
                feats = [
                    torch.cat([x[i] for x in feats]) for i in range(len_tuple)
                ]
                feats = tuple(feats)
            else:
                feats = torch.cat(feats)
        else:
            feats = self.extract_feat(inputs)
            if self.with_neck:
                feats, _ = self.neck(feats)

        if self.feature_extraction:
            feat_dim = len(feats[0].size()) if isinstance(feats, tuple) else len(
                feats.size())
            assert feat_dim in [
                5, 2
            ], ('Got feature of unknown architecture, '
                'only 3D-CNN-like ([N, in_channels, T, H, W]), and '
                'transformer-like ([N, in_channels]) features are supported.')
            if feat_dim == 5:  # 3D-CNN architecture
                # perform spatio-temporal pooling
                avg_pool = nn.AdaptiveAvgPool3d(1)
                if isinstance(feats, tuple):
                    feats = [avg_pool(x) for x in feats]
                    # concat them
                    feats = torch.cat(feats, axis=1)
                else:
                    feats = avg_pool(feats)
                # squeeze dimensions
                feats = feats.reshape((batches, num_segs, -1))
                # temporal average pooling
                feats = feats.mean(axis=1)
            return feats

        assert self.with_cls_head, 'cls head must be implemented.'
        return self.cls_head.predict(feats, data_samples)
