import torch
import torch.nn as nn

from mmaction.models.heads import I3DHead, TSNHead


def test_i3d_head():
    """Test loss method, layer construction, attributes and forward function
     in i3d head."""
    self = I3DHead(num_classes=4, in_channels=2048)
    self.init_weights()

    assert self.num_classes == 4
    assert self.dropout_ratio == 0.5
    assert self.in_channels == 2048
    assert self.init_std == 0.01

    assert isinstance(self.dropout, nn.Dropout)
    assert self.dropout.p == self.dropout_ratio

    assert isinstance(self.fc_cls, nn.Linear)
    assert self.fc_cls.in_features == self.in_channels
    assert self.fc_cls.out_features == self.num_classes

    assert isinstance(self.avg_pool, nn.AdaptiveAvgPool3d)
    assert self.avg_pool.output_size == (1, 1, 1)

    input_shape = (30, 2048, 4, 7, 7)
    feat = torch.rand(input_shape)

    cls_scores = self(feat)
    assert cls_scores.shape == torch.Size([30, 4])

    # When truth is non-empty then cls loss should be nonzero for random inputs
    gt_labels = torch.LongTensor([2] * 30).squeeze()
    losses = self.loss(cls_scores, gt_labels)
    assert 'loss_cls' in losses.keys()
    assert losses.get('loss_cls') > 0, 'cls loss should be non-zero'


def test_tsn_head():
    """Test loss method, layer construction, attributes and forward function
     in tsn head."""
    self = TSNHead(num_classes=4, in_channels=2048)
    self.init_weights()

    assert self.num_classes == 4
    assert self.dropout_ratio == 0.4
    assert self.in_channels == 2048
    assert self.init_std == 0.01
    assert self.consensus.dim == 1

    assert isinstance(self.dropout, nn.Dropout)
    assert self.dropout.p == self.dropout_ratio

    assert isinstance(self.fc_cls, nn.Linear)
    assert self.fc_cls.in_features == self.in_channels
    assert self.fc_cls.out_features == self.num_classes

    assert isinstance(self.avg_pool, nn.AdaptiveAvgPool2d)
    assert self.avg_pool.output_size == (1, 1)

    input_shape = (250, 2048, 7, 7)
    feat = torch.rand(input_shape)

    num_segs = input_shape[0]
    cls_scores = self(feat, num_segs)
    assert cls_scores.shape == torch.Size([1, 4])

    # When truth is non-empty then cls loss should be nonzero for random inputs
    gt_labels = torch.LongTensor([2])
    losses = self.loss(cls_scores, gt_labels)
    assert 'loss_cls' in losses.keys()
    assert losses.get('loss_cls') > 0, 'cls loss should be non-zero'
