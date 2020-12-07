import torch
import torch.nn as nn
from mmcv.ops import RoIAlign, RoIPool
from mmdet.models import ROI_EXTRACTORS


# register the module in mmdet
@ROI_EXTRACTORS.register_module
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
                 with_temporal_pool=True):
        super().__init__()
        self.roi_layer_type = roi_layer_type
        assert self.roi_layer_type in ['RoIPool', 'RoIAlign']
        self.featmap_stride = featmap_stride
        self.spatial_scale = 1. / self.featmap_stride

        self.output_size = output_size
        self.sampling_ratio = sampling_ratio
        self.pool_mode = pool_mode
        self.aligned = aligned

        # The attribute is needed by roi_head
        self.num_inputs = 1

        self.with_temporal_pool = with_temporal_pool
        if self.roi_layer_type == 'RoIPool':
            self.roi_layer = RoIPool(self.output_size, self.spatial_scale)
        else:
            self.roi_layer = RoIAlign(
                self.output_size,
                self.spatial_scale,
                sampling_ratio=self.sampling_ratio,
                pool_mode=self.pool_mode,
                aligned=self.aligned)

    def init_weights(self):
        pass

    def forward(self, feat, rois):
        if isinstance(feat, tuple):
            feat = feat[0]

        if self.with_temporal_pool:
            feat = torch.mean(feat, 2, keepdim=True)
        roi_feats = []
        for t in range(feat.size(2)):
            frame_feat = feat[:, :, t, :, :].contiguous()
            roi_feats.append(self.roi_layer(frame_feat, rois))
        return torch.stack(roi_feats, dim=2)
