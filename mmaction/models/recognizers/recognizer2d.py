# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from mmaction.registry import MODELS
from mmaction.utils import SampleList
from .base import BaseRecognizer


@MODELS.register_module()
class Recognizer2D(BaseRecognizer):
    """2D recognizer model framework."""

    def extract_feat(self,
                     inputs: torch.Tensor,
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

        def forward_once(batch_imgs):
            # Extract features through backbone.
            if (hasattr(self.backbone, 'features')
                    and self.backbone_from == 'torchvision'):
                x = self.backbone.features(batch_imgs)
            elif self.backbone_from == 'timm':
                x = self.backbone.forward_features(batch_imgs)
            elif self.backbone_from in ['mmcls', 'mmpretrain']:
                x = self.backbone(batch_imgs)
                if isinstance(x, tuple):
                    assert len(x) == 1
                    x = x[0]
            else:
                x = self.backbone(batch_imgs)

            if self.backbone_from in ['torchvision', 'timm']:
                if not self.feature_shape:
                    # Transformer-based feature shape: B x L x C.
                    if len(x.shape) == 3:
                        self.feature_shape = 'NLC'
                    # Resnet-based feature shape: B x C x Hs x Ws.
                    elif len(x.shape) == 4:
                        self.feature_shape = 'NCHW'

                if self.feature_shape == 'NHWC':
                    x = nn.AdaptiveAvgPool2d(1)(x.permute(0, 3, 1,
                                                          2))  # B x C x 1 x 1
                elif self.feature_shape == 'NCHW':
                    x = nn.AdaptiveAvgPool2d(1)(x)  # B x C x 1 x 1
                elif self.feature_shape == 'NLC':
                    x = nn.AdaptiveAvgPool1d(1)(x.transpose(1, 2))  # B x C x 1

                x = x.reshape((x.shape[0], -1))  # B x C
                x = x.reshape(x.shape + (1, 1))  # B x C x 1 x 1
            return x

        # Check settings of `fcn_test`.
        fcn_test = False
        if test_mode:
            if self.test_cfg is not None and self.test_cfg.get(
                    'fcn_test', False):
                fcn_test = True
                num_segs = self.test_cfg.get('num_segs',
                                             self.backbone.num_segments)
            loss_predict_kwargs['fcn_test'] = fcn_test

            # inference with batch size of `max_testing_views` if set
            if self.test_cfg is not None and self.test_cfg.get(
                    'max_testing_views', False):
                max_testing_views = self.test_cfg.get('max_testing_views')
                assert isinstance(max_testing_views, int)
                # backbone specify num_segments
                num_segments = self.backbone.get('num_segments')
                if num_segments is not None:
                    assert max_testing_views % num_segments == 0, \
                        'make sure that max_testing_views is a multiple of ' \
                        'num_segments, but got {max_testing_views} and '\
                        '{num_segments}'

                total_views = inputs.shape[0]
                view_ptr = 0
                feats = []
                while view_ptr < total_views:
                    batch_imgs = inputs[view_ptr:view_ptr + max_testing_views]
                    feat = forward_once(batch_imgs)
                    if self.with_neck:
                        feat, _ = self.neck(feat)
                    feats.append(feat)
                    view_ptr += max_testing_views

                def recursively_cat(feats):
                    # recursively traverse feats until it's a tensor,
                    # then concat
                    out_feats = []
                    for e_idx, elem in enumerate(feats[0]):
                        batch_elem = [feat[e_idx] for feat in feats]
                        if not isinstance(elem, torch.Tensor):
                            batch_elem = recursively_cat(batch_elem)
                        else:
                            batch_elem = torch.cat(batch_elem)
                        out_feats.append(batch_elem)

                    return tuple(out_feats)

                if isinstance(feats[0], tuple):
                    x = recursively_cat(feats)
                else:
                    x = torch.cat(feats)
            else:
                x = forward_once(inputs)
        else:
            x = forward_once(inputs)

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
