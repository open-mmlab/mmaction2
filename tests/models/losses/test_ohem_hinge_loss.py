# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch
from numpy.testing import assert_array_almost_equal
from torch.autograd import Variable

from mmaction.models import OHEMHingeLoss


def test_ohem_hinge_loss():
    # test normal case
    pred = torch.tensor([[
        0.5161, 0.5228, 0.7748, 0.0573, 0.1113, 0.8862, 0.1752, 0.9448, 0.0253,
        0.1009, 0.4371, 0.2232, 0.0412, 0.3487, 0.3350, 0.9294, 0.7122, 0.3072,
        0.2942, 0.7679
    ]],
                        requires_grad=True)
    gt = torch.tensor([8])
    num_video = 1
    loss = OHEMHingeLoss.apply(pred, gt, 1, 1.0, num_video)
    assert_array_almost_equal(
        loss.detach().numpy(), np.array([0.0552]), decimal=4)
    loss.backward(Variable(torch.ones([1])))
    assert_array_almost_equal(
        np.array(pred.grad),
        np.array([[
            0., 0., 0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0.
        ]]),
        decimal=4)

    # test error case
    with pytest.raises(ValueError):
        gt = torch.tensor([8, 10])
        loss = OHEMHingeLoss.apply(pred, gt, 1, 1.0, num_video)
