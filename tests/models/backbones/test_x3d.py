# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm

from mmaction.models import X3D
from mmaction.testing import check_norm_state, generate_backbone_demo_inputs


def test_x3d_backbone():
    """Test x3d backbone."""
    with pytest.raises(AssertionError):
        # In X3D: 1 <= num_stages <= 4
        X3D(gamma_w=1.0, gamma_b=2.25, gamma_d=2.2, num_stages=0)

    with pytest.raises(AssertionError):
        # In X3D: 1 <= num_stages <= 4
        X3D(gamma_w=1.0, gamma_b=2.25, gamma_d=2.2, num_stages=5)

    with pytest.raises(AssertionError):
        # len(spatial_strides) == num_stages
        X3D(gamma_w=1.0,
            gamma_b=2.25,
            gamma_d=2.2,
            spatial_strides=(1, 2),
            num_stages=4)

    with pytest.raises(AssertionError):
        # se_style in ['half', 'all']
        X3D(gamma_w=1.0, gamma_b=2.25, gamma_d=2.2, se_style=None)

    with pytest.raises(AssertionError):
        # se_ratio should be None or > 0
        X3D(gamma_w=1.0,
            gamma_b=2.25,
            gamma_d=2.2,
            se_style='half',
            se_ratio=0)

    # x3d_s, no pretrained, norm_eval True
    x3d_s = X3D(gamma_w=1.0, gamma_b=2.25, gamma_d=2.2, norm_eval=True)
    x3d_s.init_weights()
    x3d_s.train()
    assert check_norm_state(x3d_s.modules(), False)

    # x3d_l, no pretrained, norm_eval True
    x3d_l = X3D(gamma_w=1.0, gamma_b=2.25, gamma_d=5.0, norm_eval=True)
    x3d_l.init_weights()
    x3d_l.train()
    assert check_norm_state(x3d_l.modules(), False)

    # x3d_s, no pretrained, norm_eval False
    x3d_s = X3D(gamma_w=1.0, gamma_b=2.25, gamma_d=2.2, norm_eval=False)
    x3d_s.init_weights()
    x3d_s.train()
    assert check_norm_state(x3d_s.modules(), True)

    # x3d_l, no pretrained, norm_eval False
    x3d_l = X3D(gamma_w=1.0, gamma_b=2.25, gamma_d=5.0, norm_eval=False)
    x3d_l.init_weights()
    x3d_l.train()
    assert check_norm_state(x3d_l.modules(), True)

    # x3d_s, no pretrained, frozen_stages, norm_eval False
    frozen_stages = 1
    x3d_s_frozen = X3D(
        gamma_w=1.0,
        gamma_b=2.25,
        gamma_d=2.2,
        norm_eval=False,
        frozen_stages=frozen_stages)

    x3d_s_frozen.init_weights()
    x3d_s_frozen.train()
    assert x3d_s_frozen.conv1_t.bn.training is False
    for param in x3d_s_frozen.conv1_s.parameters():
        assert param.requires_grad is False
    for param in x3d_s_frozen.conv1_t.parameters():
        assert param.requires_grad is False

    for i in range(1, frozen_stages + 1):
        layer = getattr(x3d_s_frozen, f'layer{i}')
        for mod in layer.modules():
            if isinstance(mod, _BatchNorm):
                assert mod.training is False
        for param in layer.parameters():
            assert param.requires_grad is False

    # test zero_init_residual, zero_init_residual is True by default
    for m in x3d_s_frozen.modules():
        if hasattr(m, 'conv3'):
            assert torch.equal(m.conv3.bn.weight,
                               torch.zeros_like(m.conv3.bn.weight))
            assert torch.equal(m.conv3.bn.bias,
                               torch.zeros_like(m.conv3.bn.bias))

    # x3d_s inference
    input_shape = (1, 3, 13, 64, 64)
    imgs = generate_backbone_demo_inputs(input_shape)
    # parrots 3dconv is only implemented on gpu
    if torch.__version__ == 'parrots':
        if torch.cuda.is_available():
            x3d_s_frozen = x3d_s_frozen.cuda()
            imgs_gpu = imgs.cuda()
            feat = x3d_s_frozen(imgs_gpu)
            assert feat.shape == torch.Size([1, 432, 13, 2, 2])
    else:
        feat = x3d_s_frozen(imgs)
        assert feat.shape == torch.Size([1, 432, 13, 2, 2])

    # x3d_m inference
    input_shape = (1, 3, 16, 96, 96)
    imgs = generate_backbone_demo_inputs(input_shape)
    # parrots 3dconv is only implemented on gpu
    if torch.__version__ == 'parrots':
        if torch.cuda.is_available():
            x3d_s_frozen = x3d_s_frozen.cuda()
            imgs_gpu = imgs.cuda()
            feat = x3d_s_frozen(imgs_gpu)
            assert feat.shape == torch.Size([1, 432, 16, 3, 3])
    else:
        feat = x3d_s_frozen(imgs)
        assert feat.shape == torch.Size([1, 432, 16, 3, 3])
