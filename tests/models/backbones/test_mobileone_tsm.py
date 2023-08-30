# Copyright (c) OpenMMLab. All rights reserved.
import os
import tempfile

import torch
from mmengine.runner import load_checkpoint, save_checkpoint
from mmengine.runner.checkpoint import _load_checkpoint_with_prefix

from mmaction.models.backbones.mobileone_tsm import MobileOneTSM
from mmaction.testing import generate_backbone_demo_inputs


def test_mobileone_tsm_backbone():
    """Test MobileOne TSM backbone."""

    from mmpretrain.models.backbones.mobileone import MobileOneBlock

    from mmaction.models.backbones.resnet_tsm import TemporalShift

    model = MobileOneTSM('s0', pretrained2d=False)
    model.init_weights()
    for cur_module in model.modules():
        if isinstance(cur_module, TemporalShift):
            # TemporalShift is a wrapper of MobileOneBlock
            assert isinstance(cur_module.net, MobileOneBlock)
            assert cur_module.num_segments == model.num_segments
            assert cur_module.shift_div == model.shift_div

    inputs = generate_backbone_demo_inputs((8, 3, 64, 64))

    feat = model(inputs)
    assert feat.shape == torch.Size([8, 1024, 2, 2])

    model = MobileOneTSM('s1', pretrained2d=False)
    feat = model(inputs)
    assert feat.shape == torch.Size([8, 1280, 2, 2])

    model = MobileOneTSM('s2', pretrained2d=False)
    feat = model(inputs)
    assert feat.shape == torch.Size([8, 2048, 2, 2])

    model = MobileOneTSM('s3', pretrained2d=False)
    feat = model(inputs)
    assert feat.shape == torch.Size([8, 2048, 2, 2])

    model = MobileOneTSM('s4', pretrained2d=False)
    feat = model(inputs)
    assert feat.shape == torch.Size([8, 2048, 2, 2])


def test_mobileone_init_weight():
    checkpoint = ('https://download.openmmlab.com/mmclassification/v0'
                  '/mobileone/mobileone-s0_8xb32_in1k_20221110-0bc94952.pth')
    # ckpt = torch.load(checkpoint)['state_dict']
    model = MobileOneTSM(
        arch='s0',
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint, prefix='backbone'))
    model.init_weights()
    ori_ckpt = _load_checkpoint_with_prefix(
        'backbone', model.init_cfg['checkpoint'], map_location='cpu')
    for name, param in model.named_parameters():
        ori_name = name.replace('.net', '')
        assert torch.allclose(param, ori_ckpt[ori_name]), \
            f'layer {name} fail to load from pretrained checkpoint'


def test_load_deploy_mobileone():
    # Test output before and load from deploy checkpoint
    model = MobileOneTSM('s0', pretrained2d=False)
    inputs = generate_backbone_demo_inputs((8, 3, 64, 64))
    tmpdir = tempfile.gettempdir()
    ckpt_path = os.path.join(tmpdir, 'ckpt.pth')
    model.switch_to_deploy()
    model.eval()
    outputs = model(inputs)

    model_deploy = MobileOneTSM('s0', pretrained2d=False, deploy=True)
    save_checkpoint(model.state_dict(), ckpt_path)
    load_checkpoint(model_deploy, ckpt_path)

    outputs_load = model_deploy(inputs)
    for feat, feat_load in zip(outputs, outputs_load):
        assert torch.allclose(feat, feat_load)
    os.remove(ckpt_path)
