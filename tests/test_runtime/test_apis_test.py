# Copyright (c) OpenMMLab. All rights reserved.
import sys
import warnings
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# TODO import test functions from mmcv and delete them from mmaction2
try:
    from mmcv.engine import (collect_results_cpu, multi_gpu_test,
                             single_gpu_test)
    pytest.skip(
        'Test functions are supported in MMCV', allow_module_level=True)
except (ImportError, ModuleNotFoundError):
    warnings.warn(
        'DeprecationWarning: single_gpu_test, multi_gpu_test, '
        'collect_results_cpu, collect_results_gpu from mmaction2 will be '
        'deprecated. Please install mmcv through master branch.')
    from mmaction.apis.test import (collect_results_cpu, multi_gpu_test,
                                    single_gpu_test)


class OldStyleModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 1)
        self.cnt = 0

    def forward(self, *args, **kwargs):
        result = [self.cnt]
        self.cnt += 1
        return result


class Model(OldStyleModel):

    def train_step(self):
        pass

    def val_step(self):
        pass


class ExampleDataset(Dataset):

    def __init__(self):
        self.index = 0
        self.eval_result = [1, 4, 3, 7, 2, -3, 4, 6]

    def __getitem__(self, idx):
        results = dict(imgs=torch.tensor([1]))
        return results

    def __len__(self):
        return len(self.eval_result)


def test_single_gpu_test():
    test_dataset = ExampleDataset()
    loader = DataLoader(test_dataset, batch_size=1)
    model = Model()

    results = single_gpu_test(model, loader)
    assert results == list(range(8))


def mock_tensor_without_cuda(*args, **kwargs):
    if 'device' not in kwargs:
        return torch.Tensor(*args)
    return torch.IntTensor(*args, device='cpu')


@patch('mmaction.apis.test.collect_results_gpu',
       Mock(return_value=list(range(8))))
@patch('mmaction.apis.test.collect_results_cpu',
       Mock(return_value=list(range(8))))
def test_multi_gpu_test():
    test_dataset = ExampleDataset()
    loader = DataLoader(test_dataset, batch_size=1)
    model = Model()

    results = multi_gpu_test(model, loader)
    assert results == list(range(8))

    results = multi_gpu_test(model, loader, gpu_collect=False)
    assert results == list(range(8))


@patch('mmcv.runner.get_dist_info', Mock(return_value=(0, 1)))
@patch('torch.distributed.broadcast', MagicMock)
@patch('torch.distributed.barrier', Mock)
@pytest.mark.skipif(
    sys.version_info[:2] == (3, 8), reason='Not for python 3.8')
def test_collect_results_cpu():

    def content_for_unittest():
        results_part = list(range(8))
        size = 8

        results = collect_results_cpu(results_part, size)
        assert results == list(range(8))

        results = collect_results_cpu(results_part, size, 'unittest')
        assert results == list(range(8))

    if not torch.cuda.is_available():
        with patch(
                'torch.full',
                Mock(
                    return_value=torch.full(
                        (512, ), 32, dtype=torch.uint8, device='cpu'))):
            with patch('torch.tensor', mock_tensor_without_cuda):
                content_for_unittest()
    else:
        content_for_unittest()
