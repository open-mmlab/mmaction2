import copy

import pytest
import torch

from mmaction.models import TPN
from .base import generate_backbone_demo_inputs


def test_tpn():
    """Test TPN backbone."""

    tpn_cfg = dict(
        in_channels=(1024, 2048),
        out_channels=1024,
        spatial_modulation_cfg=dict(
            in_channels=(1024, 2048), out_channels=2048),
        temporal_modulation_cfg=dict(downsample_scales=(8, 8)),
        upsample_cfg=dict(scale_factor=(1, 1, 1)),
        downsample_cfg=dict(downsample_scale=(1, 1, 1)),
        level_fusion_cfg=dict(
            in_channels=(1024, 1024),
            mid_channels=(1024, 1024),
            out_channels=2048,
            downsample_scales=((1, 1, 1), (1, 1, 1))),
        aux_head_cfg=dict(out_channels=400, loss_weight=0.5))

    with pytest.raises(AssertionError):
        tpn_cfg_ = copy.deepcopy(tpn_cfg)
        tpn_cfg_['in_channels'] = list(tpn_cfg_['in_channels'])
        TPN(**tpn_cfg_)

    with pytest.raises(AssertionError):
        tpn_cfg_ = copy.deepcopy(tpn_cfg)
        tpn_cfg_['out_channels'] = float(tpn_cfg_['out_channels'])
        TPN(**tpn_cfg_)

    with pytest.raises(AssertionError):
        tpn_cfg_ = copy.deepcopy(tpn_cfg)
        tpn_cfg_['downsample_cfg']['downsample_position'] = 'unsupport'
        TPN(**tpn_cfg_)

    for k in tpn_cfg:
        if not k.endswith('_cfg'):
            continue
        tpn_cfg_ = copy.deepcopy(tpn_cfg)
        tpn_cfg_[k] = list()
        with pytest.raises(AssertionError):
            TPN(**tpn_cfg_)

    with pytest.raises(ValueError):
        tpn_cfg_ = copy.deepcopy(tpn_cfg)
        tpn_cfg_['flow_type'] = 'unsupport'
        TPN(**tpn_cfg_)

    target_shape = (32, 1)
    target = generate_backbone_demo_inputs(target_shape).long().squeeze()
    x0_shape = (32, 1024, 1, 4, 4)
    x1_shape = (32, 2048, 1, 2, 2)
    x0 = generate_backbone_demo_inputs(x0_shape)
    x1 = generate_backbone_demo_inputs(x1_shape)
    x = [x0, x1]

    # ResNetTPN with 'cascade' flow_type
    tpn_cfg_ = copy.deepcopy(tpn_cfg)
    tpn_cascade = TPN(**tpn_cfg_)
    feat, loss_aux = tpn_cascade(x, target)
    assert feat.shape == torch.Size([32, 2048, 1, 2, 2])
    assert len(loss_aux) == 1

    # ResNetTPN with 'parallel' flow_type
    tpn_cfg_ = copy.deepcopy(tpn_cfg)
    tpn_parallel = TPN(flow_type='parallel', **tpn_cfg_)
    feat, loss_aux = tpn_parallel(x, target)
    assert feat.shape == torch.Size([32, 2048, 1, 2, 2])
    assert len(loss_aux) == 1

    # ResNetTPN with 'cascade' flow_type and target is None
    feat, loss_aux = tpn_cascade(x, None)
    assert feat.shape == torch.Size([32, 2048, 1, 2, 2])
    assert len(loss_aux) == 0

    # ResNetTPN with 'parallel' flow_type and target is None
    feat, loss_aux = tpn_parallel(x, None)
    assert feat.shape == torch.Size([32, 2048, 1, 2, 2])
    assert len(loss_aux) == 0
