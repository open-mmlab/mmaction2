# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmaction.models import RGBPoseConv3D
from mmaction.testing import generate_backbone_demo_inputs


def test_rgbposeconv3d():
    """Test RGBPoseConv3D backbone."""

    with pytest.raises(AssertionError):
        RGBPoseConv3D(pose_drop_path=1.1, rgb_drop_path=1.1)

    rgbposec3d = RGBPoseConv3D()
    rgbposec3d.init_weights()
    rgbposec3d.train()

    imgs_shape = (1, 3, 8, 224, 224)
    heatmap_imgs_shape = (1, 17, 32, 56, 56)
    imgs = generate_backbone_demo_inputs(imgs_shape)
    heatmap_imgs = generate_backbone_demo_inputs(heatmap_imgs_shape)

    (x_rgb, x_pose) = rgbposec3d(imgs, heatmap_imgs)

    assert x_rgb.shape == torch.Size([1, 2048, 8, 7, 7])
    assert x_pose.shape == torch.Size([1, 512, 32, 7, 7])
