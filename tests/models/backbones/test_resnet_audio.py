# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmaction.models import ResNetAudio
from mmaction.testing import generate_backbone_demo_inputs
from mmaction.utils import register_all_modules


def test_resnet_audio_backbone():
    """Test ResNetAudio backbone."""
    input_shape = (1, 1, 16, 16)
    spec = generate_backbone_demo_inputs(input_shape)
    # inference
    register_all_modules()
    audioonly = ResNetAudio(50, None)
    audioonly.init_weights()
    audioonly.train()
    feat = audioonly(spec)
    assert feat.shape == torch.Size([1, 1024, 2, 2])
