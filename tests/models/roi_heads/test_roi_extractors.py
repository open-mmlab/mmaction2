# Copyright (c) OpenMMLab. All rights reserved.
"""import torch TODO! from mmaction.models import SingleRoIExtractor3D.

def test_single_roi_extractor3d():
    roi_extractor = SingleRoIExtractor3D(
        roi_layer_type='RoIAlign',
        featmap_stride=16,
        output_size=8,
        sampling_ratio=0,
        pool_mode='avg',
        aligned=True,
        with_temporal_pool=True)
    feat = torch.randn([4, 64, 8, 16, 16])
    rois = torch.tensor([[0., 1., 1., 6., 6.], [1., 2., 2., 7., 7.],
                         [3., 2., 2., 9., 9.], [2., 2., 0., 10., 9.]])
    roi_feat, feat = roi_extractor(feat, rois)
    assert roi_feat.shape == (4, 64, 1, 8, 8)
    assert feat.shape == (4, 64, 1, 16, 16)

    feat = (torch.randn([4, 64, 8, 16, 16]), torch.randn([4, 32, 16, 16, 16]))
    roi_feat, feat = roi_extractor(feat, rois)
    assert roi_feat.shape == (4, 96, 1, 8, 8)
    assert feat.shape == (4, 96, 1, 16, 16)

    feat = torch.randn([4, 64, 8, 16, 16])
    roi_extractor = SingleRoIExtractor3D(
        roi_layer_type='RoIAlign',
        featmap_stride=16,
        output_size=8,
        sampling_ratio=0,
        pool_mode='avg',
        aligned=True,
        with_temporal_pool=False)
    roi_feat, feat = roi_extractor(feat, rois)
    assert roi_feat.shape == (4, 64, 8, 8, 8)
    assert feat.shape == (4, 64, 8, 16, 16)

    feat = (torch.randn([4, 64, 8, 16, 16]), torch.randn([4, 32, 16, 16, 16]))
    roi_feat, feat = roi_extractor(feat, rois)
    assert roi_feat.shape == (4, 96, 16, 8, 8)
    assert feat.shape == (4, 96, 16, 16, 16)

    feat = torch.randn([4, 64, 8, 16, 16])
    roi_extractor = SingleRoIExtractor3D(
        roi_layer_type='RoIAlign',
        featmap_stride=16,
        output_size=8,
        sampling_ratio=0,
        pool_mode='avg',
        aligned=True,
        with_temporal_pool=True,
        with_global=True)
    roi_feat, feat = roi_extractor(feat, rois)
    assert roi_feat.shape == (4, 128, 1, 8, 8)
    assert feat.shape == (4, 64, 1, 16, 16)
"""
