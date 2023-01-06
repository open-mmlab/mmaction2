# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torchvision

from mmaction.models import OmniResNet
from mmaction.testing import generate_backbone_demo_inputs


def test_x3d_backbone():
    """Test x3d backbone."""
    _ = OmniResNet()

    resnet50 = torchvision.models.resnet50()
    params = resnet50.state_dict()
    torch.save(params, './r50.pth')
    model = OmniResNet(pretrain_2d='./r50.pth')

    input_shape = (2, 3, 8, 64, 64)
    videos = generate_backbone_demo_inputs(input_shape)
    feat = model(videos)
    assert feat.shape == torch.Size([2, 2048, 8, 2, 2])

    input_shape = (2, 3, 64, 64)
    images = generate_backbone_demo_inputs(input_shape)
    feat = model(images)
    assert feat.shape == torch.Size([2, 2048, 2, 2])
