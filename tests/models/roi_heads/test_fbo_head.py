# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import torch

from mmaction.models import FBOHead


def test_fbo_head():
    """Test layer construction, attributes and forward function in fbo head."""
    lfb_prefix_path = osp.normpath(
        osp.join(osp.dirname(__file__), '../../data/lfb'))

    st_feat_shape = (1, 16, 1, 8, 8)
    st_feat = torch.rand(st_feat_shape)
    rois = torch.randn(1, 5)
    rois[0][0] = 0
    img_metas = [dict(img_key='video_1, 930')]

    # non local fbo
    fbo_head = FBOHead(
        lfb_cfg=dict(
            lfb_prefix_path=lfb_prefix_path,
            max_num_sampled_feat=5,
            window_size=60,
            lfb_channels=16,
            dataset_modes=('unittest'),
            device='cpu'),
        fbo_cfg=dict(
            type='non_local',
            st_feat_channels=16,
            lt_feat_channels=16,
            latent_channels=8,
            num_st_feat=1,
            num_lt_feat=5 * 60,
        ))
    fbo_head.init_weights()
    out = fbo_head(st_feat, rois, img_metas)
    assert out.shape == (1, 24, 1, 1, 1)

    # avg fbo
    fbo_head = FBOHead(
        lfb_cfg=dict(
            lfb_prefix_path=lfb_prefix_path,
            max_num_sampled_feat=5,
            window_size=60,
            lfb_channels=16,
            dataset_modes=('unittest'),
            device='cpu'),
        fbo_cfg=dict(type='avg'))
    fbo_head.init_weights()
    out = fbo_head(st_feat, rois, img_metas)
    assert out.shape == (1, 32, 1, 1, 1)

    # max fbo
    fbo_head = FBOHead(
        lfb_cfg=dict(
            lfb_prefix_path=lfb_prefix_path,
            max_num_sampled_feat=5,
            window_size=60,
            lfb_channels=16,
            dataset_modes=('unittest'),
            device='cpu'),
        fbo_cfg=dict(type='max'))
    fbo_head.init_weights()
    out = fbo_head(st_feat, rois, img_metas)
    assert out.shape == (1, 32, 1, 1, 1)
