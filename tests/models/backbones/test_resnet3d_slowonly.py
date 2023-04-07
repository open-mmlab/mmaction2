# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmaction.models import ResNet3dSlowOnly
from mmaction.testing import generate_backbone_demo_inputs


def test_slowonly_backbone():
    """Test SlowOnly backbone."""
    with pytest.raises(AssertionError):
        # SlowOnly should contain no lateral connection
        ResNet3dSlowOnly(depth=50, pretrained=None, lateral=True)

    # test SlowOnly for PoseC3D
    so_50 = ResNet3dSlowOnly(
        depth=50,
        pretrained=None,
        in_channels=17,
        base_channels=32,
        num_stages=3,
        out_indices=(2, ),
        stage_blocks=(4, 6, 3),
        conv1_stride_s=1,
        pool1_stride_s=1,
        inflate=(0, 1, 1),
        spatial_strides=(2, 2, 2),
        temporal_strides=(1, 1, 2),
        dilations=(1, 1, 1))
    so_50.init_weights()
    so_50.train()

    # test SlowOnly with normal config
    so_50 = ResNet3dSlowOnly(depth=50, pretrained=None)
    so_50.init_weights()
    so_50.train()

    # SlowOnly inference test
    input_shape = (1, 3, 8, 64, 64)
    imgs = generate_backbone_demo_inputs(input_shape)
    # parrots 3dconv is only implemented on gpu
    if torch.__version__ == 'parrots':
        if torch.cuda.is_available():
            so_50 = so_50.cuda()
            imgs_gpu = imgs.cuda()
            feat = so_50(imgs_gpu)
    else:
        feat = so_50(imgs)
    assert feat.shape == torch.Size([1, 2048, 8, 2, 2])
