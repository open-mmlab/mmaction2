import numpy as np
import torch
import torch.nn as nn

from mmaction.models import (AudioTSNHead, BaseHead, BBoxHeadAVA, I3DHead,
                             SlowFastHead, TPNHead, TSMHead, TSNHead, X3DHead)


class ExampleHead(BaseHead):
    # use an ExampleHead to test BaseHead
    def init_weights(self):
        pass

    def forward(self, x):
        pass


def test_base_head():
    head = ExampleHead(3, 400, dict(type='CrossEntropyLoss'))

    cls_scores = torch.rand((3, 4))
    # When truth is non-empty then cls loss should be nonzero for random inputs
    gt_labels = torch.LongTensor([2] * 3).squeeze()
    losses = head.loss(cls_scores, gt_labels)
    assert 'loss_cls' in losses.keys()
    assert losses.get('loss_cls') > 0, 'cls loss should be non-zero'

    head = ExampleHead(3, 400, dict(type='CrossEntropyLoss', loss_weight=2.0))

    cls_scores = torch.rand((3, 4))
    # When truth is non-empty then cls loss should be nonzero for random inputs
    gt_labels = torch.LongTensor([2] * 3).squeeze()
    losses = head.loss(cls_scores, gt_labels)
    assert 'loss_cls' in losses.keys()
    assert losses.get('loss_cls') > 0, 'cls loss should be non-zero'


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
    bbox_head = BBoxHeadAVA()
    bbox_head.init_weights()
    bbox_head = BBoxHeadAVA(temporal_pool_type='max', spatial_pool_type='avg')
    bbox_head.init_weights()

    bbox_head = BBoxHeadAVA(in_channels=10, num_classes=4)
    input = torch.randn([3, 10, 2, 2, 2])
    ret, _ = bbox_head(input)
    assert ret.shape == (3, 4)

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
