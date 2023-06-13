# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
import torch.nn as nn
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm

from mmaction.models import ResNet3dCSN
from mmaction.testing import generate_backbone_demo_inputs


def test_resnet_csn_backbone():
    """Test resnet_csn backbone."""
    with pytest.raises(ValueError):
        # Bottleneck mode must be "ip" or "ir"
        ResNet3dCSN(152, None, bottleneck_mode='id')

    input_shape = (2, 3, 6, 64, 64)
    imgs = generate_backbone_demo_inputs(input_shape)

    resnet3d_csn_frozen = ResNet3dCSN(
        152, None, bn_frozen=True, norm_eval=True)
    resnet3d_csn_frozen.train()
    for m in resnet3d_csn_frozen.modules():
        if isinstance(m, _BatchNorm):
            for param in m.parameters():
                assert param.requires_grad is False

    # Interaction-preserved channel-separated bottleneck block
    resnet3d_csn_ip = ResNet3dCSN(152, None, bottleneck_mode='ip')
    resnet3d_csn_ip.init_weights()
    resnet3d_csn_ip.train()
    for i, layer_name in enumerate(resnet3d_csn_ip.res_layers):
        layers = getattr(resnet3d_csn_ip, layer_name)
        num_blocks = resnet3d_csn_ip.stage_blocks[i]
        assert len(layers) == num_blocks
        for layer in layers:
            assert isinstance(layer.conv2, nn.Sequential)
            assert len(layer.conv2) == 2
            assert layer.conv2[1].groups == layer.planes
    if torch.__version__ == 'parrots':
        if torch.cuda.is_available():
            resnet3d_csn_ip = resnet3d_csn_ip.cuda()
            imgs_gpu = imgs.cuda()
            feat = resnet3d_csn_ip(imgs_gpu)
            assert feat.shape == torch.Size([2, 2048, 1, 2, 2])
    else:
        feat = resnet3d_csn_ip(imgs)
        assert feat.shape == torch.Size([2, 2048, 1, 2, 2])

    # Interaction-reduced channel-separated bottleneck block
    resnet3d_csn_ir = ResNet3dCSN(152, None, bottleneck_mode='ir')
    resnet3d_csn_ir.init_weights()
    resnet3d_csn_ir.train()
    for i, layer_name in enumerate(resnet3d_csn_ir.res_layers):
        layers = getattr(resnet3d_csn_ir, layer_name)
        num_blocks = resnet3d_csn_ir.stage_blocks[i]
        assert len(layers) == num_blocks
        for layer in layers:
            assert isinstance(layer.conv2, nn.Sequential)
            assert len(layer.conv2) == 1
            assert layer.conv2[0].groups == layer.planes
    if torch.__version__ == 'parrots':
        if torch.cuda.is_available():
            resnet3d_csn_ir = resnet3d_csn_ir.cuda()
            imgs_gpu = imgs.cuda()
            feat = resnet3d_csn_ir(imgs_gpu)
            assert feat.shape == torch.Size([2, 2048, 1, 2, 2])
    else:
        feat = resnet3d_csn_ir(imgs)
        assert feat.shape == torch.Size([2, 2048, 1, 2, 2])

    # Set training status = False
    resnet3d_csn_ip = ResNet3dCSN(152, None, bottleneck_mode='ip')
    resnet3d_csn_ip.init_weights()
    resnet3d_csn_ip.train(False)
    for module in resnet3d_csn_ip.children():
        assert module.training is False
