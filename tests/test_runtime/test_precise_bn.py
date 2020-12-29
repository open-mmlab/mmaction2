import copy

import pytest
import torch
import torch.nn as nn
from mmcv.runner import EpochBasedRunner, build_optimizer
from mmcv.utils import get_logger
from torch.utils.data import DataLoader, Dataset

from mmaction.utils import PreciseBNHook


class ExampleDataset(Dataset):

    def __init__(self):
        self.index = 0

    def __getitem__(self, idx):
        results = dict(imgs=torch.tensor([1]))
        return results

    def __len__(self):
        return 1


class BiggerDataset(ExampleDataset):

    def __len__(self):
        # a bigger dataset
        return 1024


class ExampleModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Linear(1, 1)
        self.bn = nn.BatchNorm2d(1)
        self.test_cfg = None

    def forward(self, imgs, return_loss=False):
        return self.bn(self.conv(imgs))

    def train_step(self, data_batch, optimizer, **kwargs):
        outputs = {
            'loss': 0.5,
            'log_vars': {
                'accuracy': 0.98
            },
            'num_samples': 1
        }
        return outputs


def test_precise_bn():
    with pytest.raises(TypeError):
        # `data_loader` must be a Pytorch DataLoader
        test_dataset = ExampleModel()
        data_loader = DataLoader(
            test_dataset,
            batch_size=2,
            sampler=None,
            num_workers=0,
            shuffle=False)
        PreciseBNHook('data_loader')

    optimizer_cfg = dict(
        type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

    test_dataset = ExampleDataset()
    loader = DataLoader(test_dataset, batch_size=2)
    model = ExampleModel()
    optimizer = build_optimizer(model, optimizer_cfg)

    data_loader = DataLoader(test_dataset, batch_size=2)
    precise_bn_loader = copy.deepcopy(data_loader)
    logger = get_logger('precise_bn')
    runner = EpochBasedRunner(
        model=model, batch_processor=None, optimizer=optimizer, logger=logger)

    with pytest.raises(AssertionError):
        # num_iters should be no larget than total
        # iters
        precise_bn_hook = PreciseBNHook(precise_bn_loader, num_iters=5)
        runner.register_hook(precise_bn_hook)
        runner.run([loader], [('train', 1)], 1)

    test_bigger_dataset = BiggerDataset()
    loader = DataLoader(test_bigger_dataset, batch_size=2)
    precise_bn_hook = PreciseBNHook(loader, num_iters=5)
    assert precise_bn_hook.num_iters == 5
    assert precise_bn_hook.interval == 1
    runner = EpochBasedRunner(
        model=model, batch_processor=None, optimizer=optimizer, logger=logger)
    runner.register_hook(precise_bn_hook)
    runner.run([loader], [('train', 1)], 1)
