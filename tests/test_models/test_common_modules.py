import pytest
import torch

from mmaction.models import Conv2plus1d, ConvAudio


def test_conv2plus1d():
    with pytest.raises(AssertionError):
        # Length of kernel size, stride and padding must be the same
        Conv2plus1d(3, 8, (2, 2))

    conv_2plus1d = Conv2plus1d(3, 8, 2)
    conv_2plus1d.init_weights()

    assert torch.equal(conv_2plus1d.bn_s.weight,
                       torch.ones_like(conv_2plus1d.bn_s.weight))
    assert torch.equal(conv_2plus1d.bn_s.bias,
                       torch.zeros_like(conv_2plus1d.bn_s.bias))

    x = torch.rand(1, 3, 8, 256, 256)
    output = conv_2plus1d(x)
    assert output.shape == torch.Size([1, 8, 7, 255, 255])


def test_conv_audio():
    conv_audio = ConvAudio(3, 8, 3)
    conv_audio.init_weights()

    x = torch.rand(1, 3, 8, 8)
    output = conv_audio(x)
    assert output.shape == torch.Size([1, 16, 8, 8])

    conv_audio_sum = ConvAudio(3, 8, 3, op='sum')
    output = conv_audio_sum(x)
    assert output.shape == torch.Size([1, 8, 8, 8])
