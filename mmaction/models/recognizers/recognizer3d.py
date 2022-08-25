# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import Tensor

from mmaction.registry import MODELS
from mmaction.utils import SampleList
from .base import BaseRecognizer


@MODELS.register_module()
class Recognizer3D(BaseRecognizer):
    """3D recognizer model framework."""

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
                    ``loss_aux``.
        """

        # Record the kwargs required by `loss` and `predict`
        loss_predict_kwargs = dict()

        num_segs = inputs.shape[1]
        # [N, num_crops, C, T, H, W] ->
        # [N * num_crops, C, T, H, W]
        # `num_crops` is calculated by:
        #   1) `twice_sample` in `SampleFrames`
        #   2) `num_sample_positions` in `DenseSampleFrames`
        #   3) `ThreeCrop/TenCrop` in `test_pipeline`
        #   4) `num_clips` in `SampleFrames` or its subclass if `clip_len != 1`
        inputs = inputs.view((-1, ) + inputs.shape[2:])

        # Check settings of test.
        if test_mode:
            if self.test_cfg is not None:
                loss_predict_kwargs['fcn_test'] = self.test_cfg.get(
                    'fcn_test', False)
            if self.test_cfg is not None and self.test_cfg.get(
                    'max_testing_views', False):
                max_testing_views = self.test_cfg.get('max_testing_views')
                assert isinstance(max_testing_views, int)

                total_views = inputs.shape[0]
                assert num_segs == total_views, (
                    'max_testing_views is only compatible '
                    'with batch_size == 1')
                view_ptr = 0
                feats = []
                while view_ptr < total_views:
                    batch_imgs = inputs[view_ptr:view_ptr + max_testing_views]
                    feat = self.backbone(batch_imgs)
                    if self.with_neck:
                        feat, _ = self.neck(feat)
                    feats.append(feat)
                    view_ptr += max_testing_views
                # should consider the case that feat is a tuple
                if isinstance(feats[0], tuple):
                    len_tuple = len(feats[0])
                    feats = [
                        torch.cat([each[i] for each in feats])
                        for i in range(len_tuple)
                    ]
                    x = tuple(feats)
                else:
                    x = torch.cat(feats)
            else:
                x = self.backbone(inputs)
                if self.with_neck:
                    x, _ = self.neck(x)

            return x, loss_predict_kwargs
        else:
            # Return features extracted through backbone.
            x = self.backbone(inputs)
            if stage == 'backbone':
                return x, loss_predict_kwargs

            loss_aux = dict()
            if self.with_neck:
                x, loss_aux = self.neck(x, data_samples=data_samples)

            # Return features extracted through neck.
            loss_predict_kwargs['loss_aux'] = loss_aux
            if stage == 'neck':
                return x, loss_predict_kwargs

            # Return raw logits through head.
            if self.with_cls_head and stage == 'head':
                x = self.cls_head(x, **loss_predict_kwargs)
                return x, loss_predict_kwargs
