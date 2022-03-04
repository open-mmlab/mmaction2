# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import pytest
import torch
import torch.nn as nn
from mmcv.utils import assert_params_all_zeros

from mmaction.models.common import (LFB, TAM, Conv2plus1d, ConvAudio,
                                    DividedSpatialAttentionWithNorm,
                                    DividedTemporalAttentionWithNorm,
                                    FFNWithNorm, SubBatchNorm3D)


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


def test_divided_temporal_attention_with_norm():
    _cfg = dict(embed_dims=768, num_heads=12, num_frames=8)
    divided_temporal_attention = DividedTemporalAttentionWithNorm(**_cfg)
    assert isinstance(divided_temporal_attention.norm, nn.LayerNorm)
    assert assert_params_all_zeros(divided_temporal_attention.temporal_fc)

    x = torch.rand(1, 1 + 8 * 14 * 14, 768)
    output = divided_temporal_attention(x)
    assert output.shape == torch.Size([1, 1 + 8 * 14 * 14, 768])


def test_divided_spatial_attention_with_norm():
    _cfg = dict(embed_dims=512, num_heads=8, num_frames=4, dropout_layer=None)
    divided_spatial_attention = DividedSpatialAttentionWithNorm(**_cfg)
    assert isinstance(divided_spatial_attention.dropout_layer, nn.Identity)
    assert isinstance(divided_spatial_attention.norm, nn.LayerNorm)

    x = torch.rand(1, 1 + 4 * 14 * 14, 512)
    output = divided_spatial_attention(x)
    assert output.shape == torch.Size([1, 1 + 4 * 14 * 14, 512])


def test_ffn_with_norm():
    _cfg = dict(
        embed_dims=256, feedforward_channels=256 * 2, norm_cfg=dict(type='LN'))
    ffn_with_norm = FFNWithNorm(**_cfg)
    assert isinstance(ffn_with_norm.norm, nn.LayerNorm)

    x = torch.rand(1, 1 + 4 * 14 * 14, 256)
    output = ffn_with_norm(x)
    assert output.shape == torch.Size([1, 1 + 4 * 14 * 14, 256])


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


def test_LFB():
    """test LFB."""
    with pytest.raises(ValueError):
        LFB(lfb_prefix_path='./_non_exist_path')

    lfb_prefix_path = osp.normpath(
        osp.join(osp.dirname(__file__), '../data/lfb'))

    with pytest.raises(AssertionError):
        LFB(lfb_prefix_path=lfb_prefix_path, dataset_modes=100)

    with pytest.raises(ValueError):
        LFB(lfb_prefix_path=lfb_prefix_path, device='ceph')

    # load on cpu
    lfb_cpu = LFB(
        lfb_prefix_path=lfb_prefix_path,
        max_num_sampled_feat=5,
        window_size=60,
        lfb_channels=16,
        dataset_modes=('unittest'),
        device='cpu')

    lt_feat_cpu = lfb_cpu['video_1,930']
    assert lt_feat_cpu.shape == (5 * 60, 16)
    assert len(lfb_cpu) == 1

    # load on lmdb
    lfb_lmdb = LFB(
        lfb_prefix_path=lfb_prefix_path,
        max_num_sampled_feat=3,
        window_size=30,
        lfb_channels=16,
        dataset_modes=('unittest'),
        device='lmdb',
        lmdb_map_size=1e6)
    lt_feat_lmdb = lfb_lmdb['video_1,930']
    assert lt_feat_lmdb.shape == (3 * 30, 16)


def test_SubBatchNorm3D():
    _cfg = dict(num_splits=2)
    num_features = 4
    sub_batchnorm_3d = SubBatchNorm3D(num_features, **_cfg)
    assert sub_batchnorm_3d.bn.num_features == num_features
    assert sub_batchnorm_3d.split_bn.num_features == num_features * 2

    assert sub_batchnorm_3d.bn.affine is False
    assert sub_batchnorm_3d.split_bn.affine is False
