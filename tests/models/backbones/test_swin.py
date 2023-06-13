# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmaction.models import SwinTransformer3D
from mmaction.testing import generate_backbone_demo_inputs


def test_swin_backbone():
    """Test swin backbone."""
    with pytest.raises(AssertionError):
        SwinTransformer3D(arch='-t')

    with pytest.raises(AssertionError):
        SwinTransformer3D(arch={'embed_dims': 96})

    with pytest.raises(AssertionError):
        SwinTransformer3D(arch={
            'embed_dims': 96,
            'depths': [2, 2, 6],
            'num_heads': [3, 6, 12, 24]
        })

    with pytest.raises(AssertionError):
        SwinTransformer3D(
            arch={
                'embed_dims': 96,
                'depths': [2, 2, 6, 2, 2],
                'num_heads': [3, 6, 12, 24, 48]
            })

    with pytest.raises(AssertionError):
        SwinTransformer3D(arch='t', out_indices=(4, ))

    with pytest.raises(TypeError):
        swin_t = SwinTransformer3D(arch='t', pretrained=[0, 1, 1])
        swin_t.init_weights()

    with pytest.raises(TypeError):
        swin_t = SwinTransformer3D(arch='t')
        swin_t.init_weights(pretrained=[0, 1, 1])

    swin_b = SwinTransformer3D(arch='b', pretrained=None, pretrained2d=False)
    swin_b.init_weights()
    swin_b.train()

    pretrained_url = 'https://download.openmmlab.com/mmaction/v1.0/' \
                     'recognition/swin/swin_tiny_patch4_window7_224.pth'

    swin_t_pre = SwinTransformer3D(
        arch='t', pretrained=pretrained_url, pretrained2d=True)
    swin_t_pre.init_weights()
    swin_t_pre.train()

    from mmengine.runner.checkpoint import _load_checkpoint
    ckpt_2d = _load_checkpoint(pretrained_url, map_location='cpu')
    state_dict = ckpt_2d['model']

    patch_embed_weight2d = state_dict['patch_embed.proj.weight'].data
    patch_embed_weight3d = swin_t_pre.patch_embed.proj.weight.data
    assert torch.equal(
        patch_embed_weight3d,
        patch_embed_weight2d.unsqueeze(2).expand_as(patch_embed_weight3d) /
        patch_embed_weight3d.shape[2])

    norm = swin_t_pre.norm3
    assert torch.equal(norm.weight.data, state_dict['norm.weight'])
    assert torch.equal(norm.bias.data, state_dict['norm.bias'])

    for name, param in swin_t_pre.named_parameters():
        if 'relative_position_bias_table' in name:
            bias2d = state_dict[name]
            assert torch.equal(
                param.data, bias2d.repeat(2 * swin_t_pre.window_size[0] - 1,
                                          1))

    frozen_stages = 1
    swin_t_frozen = SwinTransformer3D(
        arch='t',
        pretrained=None,
        pretrained2d=False,
        frozen_stages=frozen_stages)
    swin_t_frozen.init_weights()
    swin_t_frozen.train()
    for param in swin_t_frozen.patch_embed.parameters():
        assert param.requires_grad is False
    for i in range(frozen_stages):
        layer = swin_t_frozen.layers[i]
        for param in layer.parameters():
            assert param.requires_grad is False

    input_shape = (1, 3, 6, 64, 64)
    imgs = generate_backbone_demo_inputs(input_shape)

    feat = swin_t_frozen(imgs)
    assert feat.shape == torch.Size([1, 768, 3, 2, 2])

    input_shape = (1, 3, 5, 63, 63)
    imgs = generate_backbone_demo_inputs(input_shape)
    feat = swin_t_frozen(imgs)
    assert feat.shape == torch.Size([1, 768, 3, 2, 2])

    swin_t_all_stages = SwinTransformer3D(arch='t', out_indices=(0, 1, 2, 3))
    feats = swin_t_all_stages(imgs)
    assert feats[0].shape == torch.Size([1, 96, 3, 16, 16])
    assert feats[1].shape == torch.Size([1, 192, 3, 8, 8])
    assert feats[2].shape == torch.Size([1, 384, 3, 4, 4])
    assert feats[3].shape == torch.Size([1, 768, 3, 2, 2])

    swin_t_all_stages_after_ds = SwinTransformer3D(
        arch='t', out_indices=(0, 1, 2, 3), out_after_downsample=True)
    feats = swin_t_all_stages_after_ds(imgs)
    assert feats[0].shape == torch.Size([1, 192, 3, 8, 8])
    assert feats[1].shape == torch.Size([1, 384, 3, 4, 4])
    assert feats[2].shape == torch.Size([1, 768, 3, 2, 2])
    assert feats[3].shape == torch.Size([1, 768, 3, 2, 2])
