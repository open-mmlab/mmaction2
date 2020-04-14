import torch
import torch.nn as nn

from mmaction.models.heads import BaseHead, I3DHead, TINHead, TSMHead, TSNHead


class ExampleHead(BaseHead):
    # use a ExampleHead to success BaseHead
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


def test_i3d_head():
    """Test loss method, layer construction, attributes and forward function
     in i3d head."""
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


def test_tsn_head():
    """Test loss method, layer construction, attributes and forward function
     in tsn head."""
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


def test_tsm_head():
    """Test loss method, layer construction, attributes and forward function
    in tsm head."""
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


def test_tin_head():
    """Test loss method, layer construction, attributes and forward function
    in tin head."""
    self = TINHead(num_classes=4, in_channels=2048)
    self.init_weights()

    assert self.num_classes == 4
    assert self.in_channels == 2048
    assert self.spatial_type == 'avg'
    assert self.dropout_ratio == 0.5
    assert self.init_std == 0.001
    assert self.is_shift is True
    assert self.temporal_pool is False
    assert self.consensus.dim == 1

    assert isinstance(self.dropout, nn.Dropout)
    assert self.dropout.p == self.dropout_ratio

    assert isinstance(self.fc_cls, nn.Linear)
    assert self.fc_cls.in_features == self.in_channels
    assert self.fc_cls.out_features == self.num_classes

    assert isinstance(self.avg_pool, nn.AdaptiveAvgPool2d)
    assert self.avg_pool.output_size == 1

    input_shape = (8, 2048, 7, 7)
    feat = torch.rand(input_shape)

    # tin head inference
    num_segs = input_shape[0]
    cls_scores = self(feat, num_segs)
    assert cls_scores.shape == torch.Size([1, 4])

    # tin head with temporal_pool True inference
    self = TINHead(num_classes=4, in_channels=2048, temporal_pool=True)
    self.init_weights()
    num_segs = input_shape[0]
    cls_scores = self(feat, num_segs)
    assert cls_scores.shape == torch.Size([2, 4])
