# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple, Union

import torch
from torch import Tensor

from mmaction.registry import MODELS
from .base import BaseHead


@MODELS.register_module()
class FeatureHead(BaseHead):
    """General head for feature extraction.

    Args:
        spatial_type (str, optional): Pooling type in spatial dimension.
            Default: 'avg'. If set to None, means keeping spatial dimension,
            and for GCN backbone, keeping last two dimension(T, V).
        temporal_type (str, optional): Pooling type in temporal dimension.
            Default: 'avg'. If set to None, meanse keeping temporal dimnsion,
            and for GCN backbone, keeping dimesion M. Please note that the
            channel order would keep same with the output of backbone,
            [N, T, C, H, W] for 2D recognizer, and [N, M, C, T, V] for GCN
            recognizer.
        backbone_name (str, optional): Backbone name to specifying special
            operations.Currently supports: `'tsm'`, `'slowfast'`, and `'gcn'`.
            Defaults to None, means take the input as normal feature.
        num_segments (int, optional): Number of frame segments for TSM
            backbone. Defaults to None.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 spatial_type: str = 'avg',
                 temporal_type: str = 'avg',
                 backbone_name: Optional[str] = None,
                 num_segments: Optional[str] = None,
                 **kwargs) -> None:
        super().__init__(None, None, **kwargs)

        self.temporal_type = temporal_type
        self.backbone_name = backbone_name
        self.num_segments = num_segments
        if spatial_type == 'avg':
            self.pool2d = torch.mean
        elif spatial_type == 'max':
            self.pool2d = torch.max
        elif spatial_type is None:
            self.pool2d = lambda x, dim: x
        else:
            raise NotImplementedError(
                f'Unsupported spatial_type {spatial_type}')

        if temporal_type == 'avg':
            self.pool1d = torch.mean
        elif temporal_type == 'max':
            self.pool1d = torch.max
        elif temporal_type is None:
            self.pool1d = lambda x, dim: x
        else:
            raise NotImplementedError(
                f'Unsupported temporal_type {temporal_type}')

    def forward(self,
                x: Tensor,
                num_segs: Optional[int] = None,
                **kwargs) -> Tensor:
        """Defines the computation performed at every call.

        Args:
            x (Tensor): The input data.
            num_segs (int): For 2D backbone. Number of segments into which
                a video is divided. Defaults to None.
        Returns:
            Tensor: The output features after pooling.
        """
        if isinstance(x, Tensor):
            n_dims = x.ndim
        elif isinstance(x, tuple):
            n_dims = x[0].ndim
            assert self.backbone_name == 'slowfast', \
                'Only support SlowFast backbone to input tuple'
        else:
            raise NotImplementedError(f'Unsupported feature type: {type(x)}')
        # For 2D backbone with spatial dimension
        if n_dims == 4:
            assert num_segs is not None
            if self.backbone_name == 'tsm':
                assert self.num_segments is not None, \
                    'Please Specify num_segments for TSM'
                num_segs = self.num_segments
            # [N, T, channels, H, W]
            x = x.view((-1, num_segs) + x.shape[1:])
            feat = self.pool1d(self.pool2d(x, dim=[-2, -1]), dim=1)

        elif n_dims == 5:
            if self.backbone_name == 'slowfast':
                x_slow, x_fast = x
                assert self.temporal_type is not None, \
                    'slowfast backbone has to pool temporal dimension'
                x_fast = self.pool1d(self.pool2d(x_fast, dim=[-2, -1]), dim=2)
                x_slow = self.pool1d(self.pool2d(x_slow, dim=[-2, -1]), dim=2)
                feat = torch.cat((x_slow, x_fast), dim=1)

            # For GCN-based backbone
            elif self.backbone_name == 'gcn':
                # N, M, C, T, V
                feat = self.pool1d(self.pool2d(x, dim=[-2, -1]), dim=1)
            # For 3D backbone with spatial dimension
            else:
                # [N, channels, T, H, W]
                feat = self.pool1d(self.pool2d(x, dim=[-2, -1]), dim=2)
        # For backbone output feature without spatial and temporal dimension
        elif n_dims == 2:
            # [N, channels]
            feat = x

        return feat

    def predict_by_feat(self, feats: Union[Tensor, Tuple[Tensor]],
                        data_samples) -> Tensor:
        """Integrate multi-view features into one tensor.

        Args:
            feats (torch.Tensor | tuple[torch.Tensor]): Features from
                upstream network.
            data_samples (list[:obj:`ActionDataSample`]): The batch
                data samples.

        Returns:
            Tensor: The integrated multi-view features.
        """
        num_segs = feats.shape[0] // len(data_samples)
        feats = self.average_clip(feats, num_segs=num_segs)

        return feats
