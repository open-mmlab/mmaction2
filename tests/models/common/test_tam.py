# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmaction.models.common import TAM


def test_TAM():
    """test TAM."""
    with pytest.raises(AssertionError):
        # alpha must be a positive integer
        TAM(16, 8, alpha=0, beta=4)

    with pytest.raises(AssertionError):
        # beta must be a positive integer
        TAM(16, 8, alpha=2, beta=0)

    with pytest.raises(AssertionError):
        # the channels number of x should be equal to self.in_channels of TAM
        tam = TAM(16, 8)
        x = torch.rand(64, 8, 112, 112)
        tam(x)

    tam = TAM(16, 8)
    x = torch.rand(32, 16, 112, 112)
    output = tam(x)
    assert output.shape == torch.Size([32, 16, 112, 112])
