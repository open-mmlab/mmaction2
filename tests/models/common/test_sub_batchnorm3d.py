# Copyright (c) OpenMMLab. All rights reserved.
from mmaction.models.common import SubBatchNorm3D


def test_SubBatchNorm3D():
    _cfg = dict(num_splits=2)
    num_features = 4
    sub_batchnorm_3d = SubBatchNorm3D(num_features, **_cfg)
    assert sub_batchnorm_3d.bn.num_features == num_features
    assert sub_batchnorm_3d.split_bn.num_features == num_features * 2

    assert sub_batchnorm_3d.bn.affine is False
    assert sub_batchnorm_3d.split_bn.affine is False
