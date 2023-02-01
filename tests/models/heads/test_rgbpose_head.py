# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from mmaction.models import RGBPoseHead


def test_rgbpose_head():
    """Test RGBPoseHead."""
    rgbpose_head = RGBPoseHead(
        num_classes=4,
        in_channels=[2048, 512],
        dropout=dict(rgb=0.51, pose=0.49))
    rgbpose_head.init_weights()

    assert rgbpose_head.num_classes == 4
    assert rgbpose_head.dropout == dict(rgb=0.51, pose=0.49)
    assert rgbpose_head.in_channels == [2048, 512]
    assert rgbpose_head.init_std == 0.01

    assert isinstance(rgbpose_head.dropout_rgb, nn.Dropout)
    assert isinstance(rgbpose_head.dropout_pose, nn.Dropout)
    assert rgbpose_head.dropout_rgb.p == rgbpose_head.dropout['rgb']
    assert rgbpose_head.dropout_pose.p == rgbpose_head.dropout['pose']

    assert isinstance(rgbpose_head.fc_rgb, nn.Linear)
    assert isinstance(rgbpose_head.fc_pose, nn.Linear)
    assert rgbpose_head.fc_rgb.in_features == rgbpose_head.in_channels[0]
    assert rgbpose_head.fc_rgb.out_features == rgbpose_head.num_classes
    assert rgbpose_head.fc_pose.in_features == rgbpose_head.in_channels[1]
    assert rgbpose_head.fc_pose.out_features == rgbpose_head.num_classes

    assert isinstance(rgbpose_head.avg_pool, nn.AdaptiveAvgPool3d)
    assert rgbpose_head.avg_pool.output_size == (1, 1, 1)

    feat_rgb = torch.rand((2, 2048, 8, 7, 7))
    feat_pose = torch.rand((2, 512, 32, 7, 7))

    cls_scores = rgbpose_head((feat_rgb, feat_pose))
    assert cls_scores['rgb'].shape == torch.Size([2, 4])
    assert cls_scores['pose'].shape == torch.Size([2, 4])
