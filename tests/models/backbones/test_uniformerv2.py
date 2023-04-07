# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmaction.models import UniFormerV2
from mmaction.testing import generate_backbone_demo_inputs


def test_uniformerv2_backbone():
    """Test uniformer backbone."""
    input_shape = (1, 3, 8, 64, 64)
    imgs = generate_backbone_demo_inputs(input_shape)

    model = UniFormerV2(
        input_resolution=64,
        patch_size=16,
        width=768,
        layers=12,
        heads=12,
        t_size=8,
        dw_reduction=1.5,
        backbone_drop_path_rate=0.,
        temporal_downsample=False,
        no_lmhra=True,
        double_lmhra=True,
        return_list=[8, 9, 10, 11],
        n_layers=4,
        n_dim=768,
        n_head=12,
        mlp_factor=4.,
        drop_path_rate=0.,
        clip_pretrained=False,
        mlp_dropout=[0.5, 0.5, 0.5, 0.5])
    model.init_weights()

    model.eval()
    assert model(imgs).shape == torch.Size([1, 768])

    # SthSth
    input_shape = (1, 3, 16, 64, 64)
    imgs = generate_backbone_demo_inputs(input_shape)

    model = UniFormerV2(
        input_resolution=64,
        patch_size=16,
        width=768,
        layers=12,
        heads=12,
        t_size=16,
        dw_reduction=1.5,
        backbone_drop_path_rate=0.,
        temporal_downsample=True,
        no_lmhra=False,
        double_lmhra=True,
        return_list=[8, 9, 10, 11],
        n_layers=4,
        n_dim=768,
        n_head=12,
        mlp_factor=4.,
        drop_path_rate=0.,
        clip_pretrained=False,
        mlp_dropout=[0.5, 0.5, 0.5, 0.5])
    model.init_weights()

    model.eval()
    assert model(imgs).shape == torch.Size([1, 768])
