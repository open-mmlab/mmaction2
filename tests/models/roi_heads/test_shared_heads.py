# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmaction.models import ACRNHead


def test_acrn_head():
    roi_feat = torch.randn(4, 16, 1, 7, 7)
    feat = torch.randn(2, 16, 1, 16, 16)
    rois = torch.Tensor([[0, 2.2268, 0.5926, 10.6142, 8.0029],
                         [0, 2.2577, 0.1519, 11.6451, 8.9282],
                         [1, 1.9874, 1.0000, 11.1585, 8.2840],
                         [1, 3.3338, 3.7166, 8.4174, 11.2785]])

    acrn_head = ACRNHead(32, 16)
    acrn_head.init_weights()
    new_feat = acrn_head(roi_feat, feat, rois)
    assert new_feat.shape == (4, 16, 1, 16, 16)

    acrn_head = ACRNHead(32, 16, stride=2)
    new_feat = acrn_head(roi_feat, feat, rois)
    assert new_feat.shape == (4, 16, 1, 8, 8)

    acrn_head = ACRNHead(32, 16, stride=2, num_convs=2)
    new_feat = acrn_head(roi_feat, feat, rois)
    assert new_feat.shape == (4, 16, 1, 8, 8)
