# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmaction.models import MobileNetV2TSM
from mmaction.testing import generate_backbone_demo_inputs


def test_mobilenetv2_tsm_backbone():
    """Test mobilenetv2_tsm backbone."""
    from mmcv.cnn import ConvModule

    from mmaction.models.backbones.mobilenet_v2 import InvertedResidual
    from mmaction.models.backbones.resnet_tsm import TemporalShift

    input_shape = (8, 3, 64, 64)
    imgs = generate_backbone_demo_inputs(input_shape)

    # mobilenetv2_tsm with width_mult = 1.0
    mobilenetv2_tsm = MobileNetV2TSM(pretrained='mmcls://mobilenet_v2')
    mobilenetv2_tsm.init_weights()
    for cur_module in mobilenetv2_tsm.modules():
        if isinstance(cur_module, InvertedResidual) and \
            len(cur_module.conv) == 3 and \
                cur_module.use_res_connect:
            assert isinstance(cur_module.conv[0], TemporalShift)
            assert cur_module.conv[0].num_segments == \
                mobilenetv2_tsm.num_segments
            assert cur_module.conv[0].shift_div == mobilenetv2_tsm.shift_div
            assert isinstance(cur_module.conv[0].net, ConvModule)

    # TSM-MobileNetV2 with widen_factor = 1.0 forword
    feat = mobilenetv2_tsm(imgs)
    assert feat.shape == torch.Size([8, 1280, 2, 2])

    # mobilenetv2 with widen_factor = 0.5 forword
    mobilenetv2_tsm_05 = MobileNetV2TSM(widen_factor=0.5, pretrained2d=False)
    mobilenetv2_tsm_05.init_weights()
    feat = mobilenetv2_tsm_05(imgs)
    assert feat.shape == torch.Size([8, 1280, 2, 2])

    # mobilenetv2 with widen_factor = 1.5 forword
    mobilenetv2_tsm_15 = MobileNetV2TSM(widen_factor=1.5, pretrained2d=False)
    mobilenetv2_tsm_15.init_weights()
    feat = mobilenetv2_tsm_15(imgs)
    assert feat.shape == torch.Size([8, 1920, 2, 2])
