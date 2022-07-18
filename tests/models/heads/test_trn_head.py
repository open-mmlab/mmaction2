# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
import torch.nn as nn

from mmaction.models import TRNHead


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
