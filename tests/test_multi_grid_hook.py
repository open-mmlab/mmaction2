import tempfile
import unittest.mock as mock

import mmcv
import pytest
import torch
import torch.nn as nn
from mmcv.runner import EpochBasedRunner, build_optimizer
from mmcv.utils import get_logger
from torch.utils.data import DataLoader, Dataset

from mmaction.core import MultiGridHook


class ExampleDataset(Dataset):

    def __init__(self):
        self.index = 0
        self.eval_result = [1, 4, 3, 7, 2, -3, 4, 6]

    def __getitem__(self, idx):
        results = dict(imgs=torch.tensor([1]))
        return results

    def __len__(self):
        return 1

    @mock.create_autospec
    def evaluate(self, results, logger=None):
        pass


class EvalDataset(ExampleDataset):

    def evaluate(self, results, logger=None):
        acc = self.eval_result[self.index]
        output = dict(acc=acc, index=self.index, score=acc)
        self.index += 1
        return output


class ExampleModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Linear(1, 1)
        self.test_cfg = None

    def forward(self, imgs, return_loss=False):
        return imgs

    def train_step(self, data_batch, optimizer, **kwargs):
        outputs = {
            'loss': 0.5,
            'log_vars': {
                'accuracy': 0.98
            },
            'num_samples': 1
        }
        return outputs


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires CUDA support')
def test_multi_grid_hook():
    with pytest.raises(AssertionError):
        # cfg must have fields 'data' and 'multi_grid'
        cfg = {'data': dict(size=224), 'unsupported': 123}
        multi_grid_hook = MultiGridHook(cfg)

    optimizer_cfg = dict(
        type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

    test_dataset = ExampleDataset()
    loader = DataLoader(test_dataset, batch_size=1)
    model = ExampleModel()
    optimizer = build_optimizer(model, optimizer_cfg)

    cfg = mmcv.Config.fromfile(
        'configs/recognition/i3d/i3d_r50_multigrid_32x2x1_100e_kinetics400_rgb.py'  # noqa: E501
    )
    multi_grid_hook = MultiGridHook(cfg)
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = get_logger('test_multi_grid')
        runner = EpochBasedRunner(
            model=model,
            batch_processor=None,
            optimizer=optimizer,
            work_dir=tmpdir,
            logger=logger)
        runner.register_hook(multi_grid_hook)
        runner.run([loader], [('train', 1)], 1)
