# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmaction.utils import import_module_error_class

try:
    from mmcv.ops import RoIAlign, RoIPool
except (ImportError, ModuleNotFoundError):

    @import_module_error_class('mmcv-full')
    class RoIAlign(nn.Module):
        pass

    @import_module_error_class('mmcv-full')
    class RoIPool(nn.Module):
        pass


try:
    from mmdet.models import ROI_EXTRACTORS
    mmdet_imported = True
except (ImportError, ModuleNotFoundError):
    mmdet_imported = False


class SingleRoIExtractor3D(nn.Module):
    """Extract RoI features from a single level feature map.

    Args:
        roi_layer_type (str): Specify the RoI layer type. Default: 'RoIAlign'.
        featmap_stride (int): Strides of input feature maps. Default: 16.
        output_size (int | tuple): Size or (Height, Width). Default: 16.
        sampling_ratio (int): number of inputs samples to take for each
            output sample. 0 to take samples densely for current models.
            Default: 0.
        pool_mode (str, 'avg' or 'max'): pooling mode in each bin.
            Default: 'avg'.
        aligned (bool): if False, use the legacy implementation in
            MMDetection. If True, align the results more perfectly.
            Default: True.
        with_temporal_pool (bool): if True, avgpool the temporal dim.
            Default: True.
        with_global (bool): if True, concatenate the RoI feature with global
            feature. Default: False.

    Note that sampling_ratio, pool_mode, aligned only apply when roi_layer_type
    is set as RoIAlign.
    """

    def __init__(self,
                 roi_layer_type='RoIAlign',
                 featmap_stride=16,
                 output_size=16,
                 sampling_ratio=0,
                 pool_mode='avg',
                 aligned=True,
                 with_temporal_pool=True,
                 temporal_pool_mode='avg',
                 with_global=False):
        super().__init__()
        self.roi_layer_type = roi_layer_type
        assert self.roi_layer_type in ['RoIPool', 'RoIAlign']
        self.featmap_stride = featmap_stride
        self.spatial_scale = 1. / self.featmap_stride

        self.output_size = output_size
        self.sampling_ratio = sampling_ratio
        self.pool_mode = pool_mode
        self.aligned = aligned

        self.with_temporal_pool = with_temporal_pool
        self.temporal_pool_mode = temporal_pool_mode

        self.with_global = with_global

        if self.roi_layer_type == 'RoIPool':
            self.roi_layer = RoIPool(self.output_size, self.spatial_scale)
        else:
            self.roi_layer = RoIAlign(
                self.output_size,
                self.spatial_scale,
                sampling_ratio=self.sampling_ratio,
                pool_mode=self.pool_mode,
                aligned=self.aligned)
        self.global_pool = nn.AdaptiveAvgPool2d(self.output_size)

    def init_weights(self):
        pass

    # The shape of feat is N, C, T, H, W
    def forward(self, feat, rois):
        if not isinstance(feat, tuple):
            feat = (feat, )

        if len(feat) >= 2:
            maxT = max([x.shape[2] for x in feat])
            max_shape = (maxT, ) + feat[0].shape[3:]
            # resize each feat to the largest shape (w. nearest)
            feat = [F.interpolate(x, max_shape).contiguous() for x in feat]

        if self.with_temporal_pool:
            if self.temporal_pool_mode == 'avg':
                feat = [torch.mean(x, 2, keepdim=True) for x in feat]
            elif self.temporal_pool_mode == 'max':
                feat = [torch.max(x, 2, keepdim=True)[0] for x in feat]
            else:
                raise NotImplementedError

        feat = torch.cat(feat, axis=1).contiguous()

        roi_feats = []
        for t in range(feat.size(2)):
            frame_feat = feat[:, :, t].contiguous()
            roi_feat = self.roi_layer(frame_feat, rois)
            if self.with_global:
                global_feat = self.global_pool(frame_feat.contiguous())
                inds = rois[:, 0].type(torch.int64)
                global_feat = global_feat[inds]
                roi_feat = torch.cat([roi_feat, global_feat], dim=1)
                roi_feat = roi_feat.contiguous()
            roi_feats.append(roi_feat)

        return torch.stack(roi_feats, dim=2), feat


if mmdet_imported:
    ROI_EXTRACTORS.register_module()(SingleRoIExtractor3D)
