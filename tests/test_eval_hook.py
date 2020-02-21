import tempfile
from unittest.mock import MagicMock

import mmcv.runner
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from mmaction.core import EvalHook


class ExampleDataset(Dataset):

    def __getitem__(self, idx):
        results = dict(imgs=torch.Tensor([1]))
        return results

    def __len__(self):
        return 1


class ExampleModel(nn.Module):

    def forward(self, imgs, return_loss=False):
        return imgs


def test_eval_hook():
    with pytest.raises(TypeError):
        dataset = [ExampleDataset()]
        EvalHook(dataset)

    test_dataset = ExampleDataset()
    test_dataset.evaluate = MagicMock(return_value=dict(test='success'))
    loader = DataLoader(test_dataset, batch_size=1)
    model = ExampleModel()
    eval_hook = EvalHook(test_dataset)

    with tempfile.TemporaryDirectory() as tmpdir:
        runner = mmcv.runner.Runner(
            model=model,
            batch_processor=lambda model, x, **kwargs: {
                'log_vars': {
                    "accuracy": 0.98
                },
                'num_samples': 1
            },
            work_dir=tmpdir)
        runner.register_hook(eval_hook)
        runner.run([loader], [('train', 1)], 1)
        test_dataset.evaluate.assert_called_with([torch.Tensor([1])],
                                                 logger=runner.logger)
