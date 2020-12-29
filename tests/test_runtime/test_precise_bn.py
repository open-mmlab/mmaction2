import copy

import pytest
import torch
import torch.nn as nn
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import EpochBasedRunner, build_optimizer
from mmcv.utils import get_logger
from torch.utils.data import DataLoader, Dataset

from mmaction.utils import PreciseBNHook


class ExampleDataset(Dataset):

    def __init__(self):
        self.index = 0

    def __getitem__(self, idx):
        results = dict(imgs=torch.tensor([1.0], dtype=torch.float32))
        return results

    def __len__(self):
        return 1


class BiggerDataset(ExampleDataset):

    def __len__(self):
        # a bigger dataset
        return 12


class ExampleModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Linear(1, 1)
        self.bn = nn.BatchNorm1d(1)
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


class GNExampleModel(ExampleModel):

    def __init__(self):
        super().__init__()
        self.conv = nn.Linear(1, 1)
        self.bn = nn.GroupNorm(1, 1)
        self.test_cfg = None


class NoBNExampleModel(ExampleModel):

    def __init__(self):
        super().__init__()
        self.conv = nn.Linear(1, 1)
        self.test_cfg = None

    def forward(self, imgs, return_loss=False):
        return self.conv(imgs)


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

    # test non-DDP model
    test_bigger_dataset = BiggerDataset()
    loader = DataLoader(test_bigger_dataset, batch_size=2)
    precise_bn_hook = PreciseBNHook(loader, num_iters=5)
    assert precise_bn_hook.num_iters == 5
    assert precise_bn_hook.interval == 1
    runner = EpochBasedRunner(
        model=model, batch_processor=None, optimizer=optimizer, logger=logger)
    runner.register_hook(precise_bn_hook)
    runner.run([loader], [('train', 1)], 1)

    # test model w/ gn layer
    loader = DataLoader(test_bigger_dataset, batch_size=2)
    precise_bn_hook = PreciseBNHook(loader, num_iters=5)
    assert precise_bn_hook.num_iters == 5
    assert precise_bn_hook.interval == 1
    model = GNExampleModel()
    runner = EpochBasedRunner(
        model=model, batch_processor=None, optimizer=optimizer, logger=logger)
    runner.register_hook(precise_bn_hook)
    runner.run([loader], [('train', 1)], 1)

    # test model without bn layer
    loader = DataLoader(test_bigger_dataset, batch_size=2)
    precise_bn_hook = PreciseBNHook(loader, num_iters=5)
    assert precise_bn_hook.num_iters == 5
    assert precise_bn_hook.interval == 1
    model = NoBNExampleModel()
    runner = EpochBasedRunner(
        model=model, batch_processor=None, optimizer=optimizer, logger=logger)
    runner.register_hook(precise_bn_hook)
    runner.run([loader], [('train', 1)], 1)

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason='requires CUDA support')
    def test_ddp_model_precise_bn():
        # test DDP model
        test_bigger_dataset = BiggerDataset()
        loader = DataLoader(test_bigger_dataset, batch_size=2)
        precise_bn_hook = PreciseBNHook(loader, num_iters=5)
        assert precise_bn_hook.num_iters == 5
        assert precise_bn_hook.interval == 1
        model = ExampleModel()
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=True)
        runner = EpochBasedRunner(
            model=model,
            batch_processor=None,
            optimizer=optimizer,
            logger=logger)
        runner.register_hook(precise_bn_hook)
        runner.run([loader], [('train', 1)], 1)
