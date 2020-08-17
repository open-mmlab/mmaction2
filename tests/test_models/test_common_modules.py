import pytest
import torch

from mmaction.models import Conv2plus1d, SubBatchBN3d


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


def test_sub_batch_bn3d():
    num_features = 64
    num_splits = 6
    # subbn3d without affine
    cfg = {'num_splits': num_splits, 'affine': False}
    subbn3d_no_affine = SubBatchBN3d(num_features, **cfg)
    # test training
    subbn3d_no_affine.train()
    assert (subbn3d_no_affine.split_bn.num_features == num_features *
            num_splits)
    x = torch.rand(12, num_features, 8, 14, 14)
    output = subbn3d_no_affine(x)
    subbn3d_no_affine.aggregate_stats()
    assert torch.equal(
        subbn3d_no_affine.bn.running_mean.data,
        (subbn3d_no_affine.split_bn.running_mean.view(num_splits, -1).sum(0) /
         num_splits))
    assert output.size() == x.size()

    # test eval
    output = subbn3d_no_affine(x)
    assert output.size() == x.size()

    # subbn3d with affine
    cfg = {'num_splits': 6}
    subbn3d_affine = SubBatchBN3d(num_features, **cfg)
    assert torch.equal(subbn3d_affine.weight,
                       torch.ones_like(subbn3d_affine.weight))
    assert torch.equal(subbn3d_affine.bias,
                       torch.zeros_like(subbn3d_affine.bias))
    output = subbn3d_affine(x)
    assert output.size() == x.size()
