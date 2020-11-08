from unittest.mock import Mock, patch

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from mmaction.apis.test import (collect_results_cpu, multi_gpu_test,
                                single_gpu_test)


class OldStyleModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 1)
        self.cnt = 0

    def forward(self, return_loss, **kwargs):
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
        print(*args)
        print(args)
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
@patch('torch.distributed.broadcast', Mock)
@patch(
    'torch.full',
    Mock(
        return_value=torch.full((512, ), 32, dtype=torch.uint8, device='cpu')))
@patch('torch.tensor', mock_tensor_without_cuda)
@patch('torch.distributed.barrier', Mock)
def test_collect_results_cpu():
    results_part = list(range(8))
    size = 8

    results = collect_results_cpu(results_part, size)
    assert results == list(range(8))

    results = collect_results_cpu(results_part, size, 'unittest')
    assert results == list(range(8))
