import tempfile
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from mmcv.runner import Runner
from mmcv.utils import get_logger
from torch.utils.data import DataLoader, Dataset

from mmaction.core import EvalHook


class ExampleDataset(Dataset):

    def __getitem__(self, idx):
        results = dict(imgs=torch.tensor([1]))
        return results

    def __len__(self):
        return 1


class ExampleModel(nn.Module):

    def __init__(self):
        super(ExampleModel, self).__init__()
        self.test_cfg = None

    def forward(self, imgs, return_loss=False):
        return imgs


def test_eval_hook():
    with pytest.raises(TypeError):
        test_dataset = ExampleModel()
        data_loader = [
            DataLoader(
                test_dataset,
                batch_size=1,
                sampler=None,
                num_worker=0,
                shuffle=False)
        ]
        EvalHook(data_loader)

    test_dataset = ExampleDataset()
    test_dataset.evaluate = MagicMock(return_value=dict(test='success'))
    loader = DataLoader(test_dataset, batch_size=1)
    model = ExampleModel()
    data_loader = DataLoader(
        test_dataset, batch_size=1, sampler=None, num_workers=0, shuffle=False)
    eval_hook = EvalHook(data_loader)

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = get_logger('test_eval')
        runner = Runner(
            model=model,
            batch_processor=lambda model, x, **kwargs: {
                'log_vars': {
                    'accuracy': 0.98
                },
                'num_samples': 1
            },
            work_dir=tmpdir,
            logger=logger)
        runner.register_hook(eval_hook)
        runner.run([loader], [('train', 1)], 1)
        test_dataset.evaluate.assert_called_with([torch.tensor([1])],
                                                 logger=runner.logger)
