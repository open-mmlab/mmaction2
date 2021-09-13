# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

import mmaction
from mmaction.models import (ACRNHead, AudioTSNHead, BBoxHeadAVA, FBOHead,
                             I3DHead, LFBInferHead, SlowFastHead, STGCNHead,
                             TimeSformerHead, TPNHead, TRNHead, TSMHead,
                             TSNHead, X3DHead)
from .base import generate_backbone_demo_inputs


def test_i3d_head():
    """Test loss method, layer construction, attributes and forward function in
    i3d head."""
    i3d_head = I3DHead(num_classes=4, in_channels=2048)
    i3d_head.init_weights()

    assert i3d_head.num_classes == 4
    assert i3d_head.dropout_ratio == 0.5
    assert i3d_head.in_channels == 2048
    assert i3d_head.init_std == 0.01

    assert isinstance(i3d_head.dropout, nn.Dropout)
    assert i3d_head.dropout.p == i3d_head.dropout_ratio

    assert isinstance(i3d_head.fc_cls, nn.Linear)
    assert i3d_head.fc_cls.in_features == i3d_head.in_channels
    assert i3d_head.fc_cls.out_features == i3d_head.num_classes

    assert isinstance(i3d_head.avg_pool, nn.AdaptiveAvgPool3d)
    assert i3d_head.avg_pool.output_size == (1, 1, 1)

    input_shape = (3, 2048, 4, 7, 7)
    feat = torch.rand(input_shape)

    # i3d head inference
    cls_scores = i3d_head(feat)
    assert cls_scores.shape == torch.Size([3, 4])


def test_bbox_head_ava():
    """Test loss method, layer construction, attributes and forward function in
    bbox head."""
    with pytest.raises(TypeError):
        # topk must be None, int or tuple[int]
        BBoxHeadAVA(topk=0.1)

    with pytest.raises(AssertionError):
        # topk should be smaller than num_classes
        BBoxHeadAVA(num_classes=5, topk=(3, 5))

    bbox_head = BBoxHeadAVA(in_channels=10, num_classes=4, topk=1)
    input = torch.randn([3, 10, 2, 2, 2])
    ret, _ = bbox_head(input)
    assert ret.shape == (3, 4)

    bbox_head = BBoxHeadAVA()
    bbox_head.init_weights()
    bbox_head = BBoxHeadAVA(temporal_pool_type='max', spatial_pool_type='avg')
    bbox_head.init_weights()

    cls_score = torch.tensor(
        [[0.568, -0.162, 0.273, -0.390, 0.447, 0.102, -0.409],
         [2.388, 0.609, 0.369, 1.630, -0.808, -0.212, 0.296],
         [0.252, -0.533, -0.644, -0.591, 0.148, 0.963, -0.525],
         [0.134, -0.311, -0.764, -0.752, 0.656, -1.517, 0.185]])
    labels = torch.tensor([[0., 0., 1., 0., 0., 1., 0.],
                           [0., 0., 0., 1., 0., 0., 0.],
                           [0., 1., 0., 0., 1., 0., 1.],
                           [0., 0., 1., 1., 0., 0., 1.]])
    label_weights = torch.tensor([1., 1., 1., 1.])
    losses = bbox_head.loss(
        cls_score=cls_score,
        bbox_pred=None,
        rois=None,
        labels=labels,
        label_weights=label_weights)
    assert torch.isclose(losses['loss_action_cls'], torch.tensor(0.7162495))
    assert torch.isclose(losses['recall@thr=0.5'], torch.tensor(0.6666666))
    assert torch.isclose(losses['prec@thr=0.5'], torch.tensor(0.4791665))
    assert torch.isclose(losses['recall@top3'], torch.tensor(0.75))
    assert torch.isclose(losses['prec@top3'], torch.tensor(0.5))
    assert torch.isclose(losses['recall@top5'], torch.tensor(1.0))
    assert torch.isclose(losses['prec@top5'], torch.tensor(0.45))

    rois = torch.tensor([[0.0, 0.1, 0.2, 0.3, 0.4], [0.0, 0.5, 0.6, 0.7, 0.8]])
    rois[1::2] *= 380
    rois[2::2] *= 220
    crop_quadruple = np.array([0.1, 0.2, 0.8, 0.7])
    cls_score = torch.tensor([0.995, 0.728])
    img_shape = (320, 480)
    flip = True

    bboxes, scores = bbox_head.get_det_bboxes(
        rois=rois,
        cls_score=cls_score,
        img_shape=img_shape,
        flip=flip,
        crop_quadruple=crop_quadruple)
    assert torch.all(
        torch.isclose(
            bboxes,
            torch.tensor([[0.89783341, 0.20043750, 0.89816672, 0.20087500],
                          [0.45499998, 0.69875002, 0.58166665, 0.86499995]])))
    assert torch.all(
        torch.isclose(scores, torch.tensor([0.73007441, 0.67436624])))


def test_x3d_head():
    """Test loss method, layer construction, attributes and forward function in
    x3d head."""
    x3d_head = X3DHead(in_channels=432, num_classes=4, fc1_bias=False)
    x3d_head.init_weights()

    assert x3d_head.num_classes == 4
    assert x3d_head.dropout_ratio == 0.5
    assert x3d_head.in_channels == 432
    assert x3d_head.init_std == 0.01

    assert isinstance(x3d_head.dropout, nn.Dropout)
    assert x3d_head.dropout.p == x3d_head.dropout_ratio

    assert isinstance(x3d_head.fc1, nn.Linear)
    assert x3d_head.fc1.in_features == x3d_head.in_channels
    assert x3d_head.fc1.out_features == x3d_head.mid_channels
    assert x3d_head.fc1.bias is None

    assert isinstance(x3d_head.fc2, nn.Linear)
    assert x3d_head.fc2.in_features == x3d_head.mid_channels
    assert x3d_head.fc2.out_features == x3d_head.num_classes

    assert isinstance(x3d_head.pool, nn.AdaptiveAvgPool3d)
    assert x3d_head.pool.output_size == (1, 1, 1)

    input_shape = (3, 432, 4, 7, 7)
    feat = torch.rand(input_shape)

    # i3d head inference
    cls_scores = x3d_head(feat)
    assert cls_scores.shape == torch.Size([3, 4])


def test_slowfast_head():
    """Test loss method, layer construction, attributes and forward function in
    slowfast head."""
    sf_head = SlowFastHead(num_classes=4, in_channels=2304)
    sf_head.init_weights()

    assert sf_head.num_classes == 4
    assert sf_head.dropout_ratio == 0.8
    assert sf_head.in_channels == 2304
    assert sf_head.init_std == 0.01

    assert isinstance(sf_head.dropout, nn.Dropout)
    assert sf_head.dropout.p == sf_head.dropout_ratio

    assert isinstance(sf_head.fc_cls, nn.Linear)
    assert sf_head.fc_cls.in_features == sf_head.in_channels
    assert sf_head.fc_cls.out_features == sf_head.num_classes

    assert isinstance(sf_head.avg_pool, nn.AdaptiveAvgPool3d)
    assert sf_head.avg_pool.output_size == (1, 1, 1)

    input_shape = (3, 2048, 32, 7, 7)
    feat_slow = torch.rand(input_shape)

    input_shape = (3, 256, 4, 7, 7)
    feat_fast = torch.rand(input_shape)

    sf_head = SlowFastHead(num_classes=4, in_channels=2304)
    cls_scores = sf_head((feat_slow, feat_fast))
    assert cls_scores.shape == torch.Size([3, 4])


def test_tsn_head():
    """Test loss method, layer construction, attributes and forward function in
    tsn head."""
    tsn_head = TSNHead(num_classes=4, in_channels=2048)
    tsn_head.init_weights()

    assert tsn_head.num_classes == 4
    assert tsn_head.dropout_ratio == 0.4
    assert tsn_head.in_channels == 2048
    assert tsn_head.init_std == 0.01
    assert tsn_head.consensus.dim == 1
    assert tsn_head.spatial_type == 'avg'

    assert isinstance(tsn_head.dropout, nn.Dropout)
    assert tsn_head.dropout.p == tsn_head.dropout_ratio

    assert isinstance(tsn_head.fc_cls, nn.Linear)
    assert tsn_head.fc_cls.in_features == tsn_head.in_channels
    assert tsn_head.fc_cls.out_features == tsn_head.num_classes

    assert isinstance(tsn_head.avg_pool, nn.AdaptiveAvgPool2d)
    assert tsn_head.avg_pool.output_size == (1, 1)

    input_shape = (8, 2048, 7, 7)
    feat = torch.rand(input_shape)

    # tsn head inference
    num_segs = input_shape[0]
    cls_scores = tsn_head(feat, num_segs)
    assert cls_scores.shape == torch.Size([1, 4])

    # Test multi-class recognition
    multi_tsn_head = TSNHead(
        num_classes=4,
        in_channels=2048,
        loss_cls=dict(type='BCELossWithLogits', loss_weight=160.0),
        multi_class=True,
        label_smooth_eps=0.01)
    multi_tsn_head.init_weights()
    assert multi_tsn_head.num_classes == 4
    assert multi_tsn_head.dropout_ratio == 0.4
    assert multi_tsn_head.in_channels == 2048
    assert multi_tsn_head.init_std == 0.01
    assert multi_tsn_head.consensus.dim == 1

    assert isinstance(multi_tsn_head.dropout, nn.Dropout)
    assert multi_tsn_head.dropout.p == multi_tsn_head.dropout_ratio

    assert isinstance(multi_tsn_head.fc_cls, nn.Linear)
    assert multi_tsn_head.fc_cls.in_features == multi_tsn_head.in_channels
    assert multi_tsn_head.fc_cls.out_features == multi_tsn_head.num_classes

    assert isinstance(multi_tsn_head.avg_pool, nn.AdaptiveAvgPool2d)
    assert multi_tsn_head.avg_pool.output_size == (1, 1)

    input_shape = (8, 2048, 7, 7)
    feat = torch.rand(input_shape)

    # multi-class tsn head inference
    num_segs = input_shape[0]
    cls_scores = tsn_head(feat, num_segs)
    assert cls_scores.shape == torch.Size([1, 4])


def test_tsn_head_audio():
    """Test loss method, layer construction, attributes and forward function in
    tsn head."""
    tsn_head_audio = AudioTSNHead(num_classes=4, in_channels=5)
    tsn_head_audio.init_weights()

    assert tsn_head_audio.num_classes == 4
    assert tsn_head_audio.dropout_ratio == 0.4
    assert tsn_head_audio.in_channels == 5
    assert tsn_head_audio.init_std == 0.01
    assert tsn_head_audio.spatial_type == 'avg'

    assert isinstance(tsn_head_audio.dropout, nn.Dropout)
    assert tsn_head_audio.dropout.p == tsn_head_audio.dropout_ratio

    assert isinstance(tsn_head_audio.fc_cls, nn.Linear)
    assert tsn_head_audio.fc_cls.in_features == tsn_head_audio.in_channels
    assert tsn_head_audio.fc_cls.out_features == tsn_head_audio.num_classes

    assert isinstance(tsn_head_audio.avg_pool, nn.AdaptiveAvgPool2d)
    assert tsn_head_audio.avg_pool.output_size == (1, 1)

    input_shape = (8, 5, 7, 7)
    feat = torch.rand(input_shape)

    # tsn head inference
    cls_scores = tsn_head_audio(feat)
    assert cls_scores.shape == torch.Size([8, 4])


def test_tsm_head():
    """Test loss method, layer construction, attributes and forward function in
    tsm head."""
    tsm_head = TSMHead(num_classes=4, in_channels=2048)
    tsm_head.init_weights()

    assert tsm_head.num_classes == 4
    assert tsm_head.dropout_ratio == 0.8
    assert tsm_head.in_channels == 2048
    assert tsm_head.init_std == 0.001
    assert tsm_head.consensus.dim == 1
    assert tsm_head.spatial_type == 'avg'

    assert isinstance(tsm_head.dropout, nn.Dropout)
    assert tsm_head.dropout.p == tsm_head.dropout_ratio

    assert isinstance(tsm_head.fc_cls, nn.Linear)
    assert tsm_head.fc_cls.in_features == tsm_head.in_channels
    assert tsm_head.fc_cls.out_features == tsm_head.num_classes

    assert isinstance(tsm_head.avg_pool, nn.AdaptiveAvgPool2d)
    assert tsm_head.avg_pool.output_size == 1

    input_shape = (8, 2048, 7, 7)
    feat = torch.rand(input_shape)

    # tsm head inference with no init
    num_segs = input_shape[0]
    cls_scores = tsm_head(feat, num_segs)
    assert cls_scores.shape == torch.Size([1, 4])

    # tsm head inference with init
    tsm_head = TSMHead(num_classes=4, in_channels=2048, temporal_pool=True)
    tsm_head.init_weights()
    cls_scores = tsm_head(feat, num_segs)
    assert cls_scores.shape == torch.Size([2, 4])


def test_trn_head():
    """Test loss method, layer construction, attributes and forward function in
    trn head."""
    from mmaction.models.heads.trn_head import (RelationModule,
                                                RelationModuleMultiScale)
    trn_head = TRNHead(num_classes=4, in_channels=2048, relation_type='TRN')
    trn_head.init_weights()

    assert trn_head.num_classes == 4
    assert trn_head.dropout_ratio == 0.8
    assert trn_head.in_channels == 2048
    assert trn_head.init_std == 0.001
    assert trn_head.spatial_type == 'avg'

    relation_module = trn_head.consensus
    assert isinstance(relation_module, RelationModule)
    assert relation_module.hidden_dim == 256
    assert isinstance(relation_module.classifier[3], nn.Linear)
    assert relation_module.classifier[3].out_features == trn_head.num_classes

    assert trn_head.dropout.p == trn_head.dropout_ratio
    assert isinstance(trn_head.dropout, nn.Dropout)
    assert isinstance(trn_head.fc_cls, nn.Linear)
    assert trn_head.fc_cls.in_features == trn_head.in_channels
    assert trn_head.fc_cls.out_features == trn_head.hidden_dim

    assert isinstance(trn_head.avg_pool, nn.AdaptiveAvgPool2d)
    assert trn_head.avg_pool.output_size == 1

    input_shape = (8, 2048, 7, 7)
    feat = torch.rand(input_shape)

    # tsm head inference with no init
    num_segs = input_shape[0]
    cls_scores = trn_head(feat, num_segs)
    assert cls_scores.shape == torch.Size([1, 4])

    # tsm head inference with init
    trn_head = TRNHead(
        num_classes=4,
        in_channels=2048,
        num_segments=8,
        relation_type='TRNMultiScale')
    trn_head.init_weights()
    assert isinstance(trn_head.consensus, RelationModuleMultiScale)
    assert trn_head.consensus.scales == range(8, 1, -1)
    cls_scores = trn_head(feat, num_segs)
    assert cls_scores.shape == torch.Size([1, 4])

    with pytest.raises(ValueError):
        trn_head = TRNHead(
            num_classes=4,
            in_channels=2048,
            num_segments=8,
            relation_type='RelationModlue')


def test_timesformer_head():
    """Test loss method, layer construction, attributes and forward function in
    timesformer head."""
    timesformer_head = TimeSformerHead(num_classes=4, in_channels=64)
    timesformer_head.init_weights()

    assert timesformer_head.num_classes == 4
    assert timesformer_head.in_channels == 64
    assert timesformer_head.init_std == 0.02

    input_shape = (2, 64)
    feat = torch.rand(input_shape)

    cls_scores = timesformer_head(feat)
    assert cls_scores.shape == torch.Size([2, 4])


@patch.object(mmaction.models.LFBInferHead, '__del__', Mock)
def test_lfb_infer_head():
    """Test layer construction, attributes and forward function in lfb infer
    head."""
    with tempfile.TemporaryDirectory() as tmpdir:
        lfb_infer_head = LFBInferHead(
            lfb_prefix_path=tmpdir, use_half_precision=True)
    lfb_infer_head.init_weights()

    st_feat_shape = (3, 16, 1, 8, 8)
    st_feat = generate_backbone_demo_inputs(st_feat_shape)
    rois = torch.cat(
        (torch.tensor([0, 1, 0]).float().view(3, 1), torch.randn(3, 4)), dim=1)
    img_metas = [dict(img_key='video_1,777'), dict(img_key='video_2, 888')]
    result = lfb_infer_head(st_feat, rois, img_metas)
    assert st_feat.equal(result)
    assert len(lfb_infer_head.all_features) == 3
    assert lfb_infer_head.all_features[0].shape == (16, 1, 1, 1)


def test_fbo_head():
    """Test layer construction, attributes and forward function in fbo head."""
    lfb_prefix_path = osp.normpath(
        osp.join(osp.dirname(__file__), '../data/lfb'))

    st_feat_shape = (1, 16, 1, 8, 8)
    st_feat = generate_backbone_demo_inputs(st_feat_shape)
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


def test_tpn_head():
    """Test loss method, layer construction, attributes and forward function in
    tpn head."""
    tpn_head = TPNHead(num_classes=4, in_channels=2048)
    tpn_head.init_weights()

    assert hasattr(tpn_head, 'avg_pool2d')
    assert hasattr(tpn_head, 'avg_pool3d')
    assert isinstance(tpn_head.avg_pool3d, nn.AdaptiveAvgPool3d)
    assert tpn_head.avg_pool3d.output_size == (1, 1, 1)
    assert tpn_head.avg_pool2d is None

    input_shape = (4, 2048, 7, 7)
    feat = torch.rand(input_shape)

    # tpn head inference with num_segs
    num_segs = 2
    cls_scores = tpn_head(feat, num_segs)
    assert isinstance(tpn_head.avg_pool2d, nn.AvgPool3d)
    assert tpn_head.avg_pool2d.kernel_size == (1, 7, 7)
    assert cls_scores.shape == torch.Size([2, 4])

    # tpn head inference with no num_segs
    input_shape = (2, 2048, 3, 7, 7)
    feat = torch.rand(input_shape)
    cls_scores = tpn_head(feat)
    assert isinstance(tpn_head.avg_pool2d, nn.AvgPool3d)
    assert tpn_head.avg_pool2d.kernel_size == (1, 7, 7)
    assert cls_scores.shape == torch.Size([2, 4])


def test_acrn_head():
    roi_feat = torch.randn(4, 16, 1, 7, 7)
    feat = torch.randn(2, 16, 1, 16, 16)
    rois = torch.Tensor([[0, 2.2268, 0.5926, 10.6142, 8.0029],
                         [0, 2.2577, 0.1519, 11.6451, 8.9282],
                         [1, 1.9874, 1.0000, 11.1585, 8.2840],
                         [1, 3.3338, 3.7166, 8.4174, 11.2785]])

    acrn_head = ACRNHead(32, 16)
    acrn_head.init_weights()
    new_feat = acrn_head(roi_feat, feat, rois)
    assert new_feat.shape == (4, 16, 1, 16, 16)

    acrn_head = ACRNHead(32, 16, stride=2)
    new_feat = acrn_head(roi_feat, feat, rois)
    assert new_feat.shape == (4, 16, 1, 8, 8)

    acrn_head = ACRNHead(32, 16, stride=2, num_convs=2)
    new_feat = acrn_head(roi_feat, feat, rois)
    assert new_feat.shape == (4, 16, 1, 8, 8)


def test_stgcn_head():
    """Test loss method, layer construction, attributes and forward function in
    stgcn head."""
    with pytest.raises(NotImplementedError):
        # spatial_type not in ['avg', 'max']
        stgcn_head = STGCNHead(
            num_classes=60, in_channels=256, spatial_type='min')
        stgcn_head.init_weights()

    # spatial_type='avg'
    stgcn_head = STGCNHead(num_classes=60, in_channels=256, spatial_type='avg')
    stgcn_head.init_weights()

    assert stgcn_head.num_classes == 60
    assert stgcn_head.in_channels == 256

    input_shape = (2, 256, 75, 17)
    feat = torch.rand(input_shape)

    cls_scores = stgcn_head(feat)
    assert cls_scores.shape == torch.Size([1, 60])

    # spatial_type='max'
    stgcn_head = STGCNHead(num_classes=60, in_channels=256, spatial_type='max')
    stgcn_head.init_weights()

    assert stgcn_head.num_classes == 60
    assert stgcn_head.in_channels == 256

    input_shape = (2, 256, 75, 17)
    feat = torch.rand(input_shape)

    cls_scores = stgcn_head(feat)
    assert cls_scores.shape == torch.Size([1, 60])
