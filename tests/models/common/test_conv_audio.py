# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmaction.models.common import ConvAudio


def test_conv_audio():
    conv_audio = ConvAudio(3, 8, 3)
    conv_audio.init_weights()

    x = torch.rand(1, 3, 8, 8)
    output = conv_audio(x)
    assert output.shape == torch.Size([1, 16, 8, 8])

    conv_audio_sum = ConvAudio(3, 8, 3, op='sum')
    output = conv_audio_sum(x)
    assert output.shape == torch.Size([1, 8, 8, 8])
