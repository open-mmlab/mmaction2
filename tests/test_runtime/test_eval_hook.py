import os.path as osp
import tempfile
import unittest.mock as mock
from collections import OrderedDict
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from mmcv.runner import EpochBasedRunner, build_optimizer
from mmcv.utils import get_logger
from torch.utils.data import DataLoader, Dataset

from mmaction.core import DistEpochEvalHook, EpochEvalHook


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
        output = OrderedDict(acc=acc, index=self.index, score=acc)
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


def _build_demo_runner():

    class Model(nn.Module):

        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 1)

        def forward(self, x):
            return self.linear(x)

        def train_step(self, x, optimizer, **kwargs):
            return dict(loss=self(x))

        def val_step(self, x, optimizer, **kwargs):
            return dict(loss=self(x))

    model = Model()
    tmp_dir = tempfile.mkdtemp()

    runner = EpochBasedRunner(
        model=model, work_dir=tmp_dir, logger=get_logger('demo'))
    return runner


def test_eval_hook():
    with pytest.raises(AssertionError):
        # `save_best` should be a str
        test_dataset = ExampleModel()
        data_loader = DataLoader(
            test_dataset,
            batch_size=1,
            sampler=None,
            num_workers=0,
            shuffle=False)
        EpochEvalHook(data_loader, save_best=True)

    with pytest.raises(TypeError):
        # dataloader must be a pytorch DataLoader
        test_dataset = ExampleModel()
        data_loader = [
            DataLoader(
                test_dataset,
                batch_size=1,
                sampler=None,
                num_worker=0,
                shuffle=False)
        ]
        EpochEvalHook(data_loader)

    with pytest.raises(ValueError):
        # key_indicator must be valid when rule_map is None
        test_dataset = ExampleDataset()
        data_loader = DataLoader(
            test_dataset,
            batch_size=1,
            sampler=None,
            num_workers=0,
            shuffle=False)
        EpochEvalHook(data_loader, save_best='unsupport')

    with pytest.raises(KeyError):
        # rule must be in keys of rule_map
        test_dataset = ExampleModel()
        data_loader = DataLoader(
            test_dataset,
            batch_size=1,
            sampler=None,
            num_workers=0,
            shuffle=False)
        EpochEvalHook(data_loader, save_best='auto', rule='unsupport')

    optimizer_cfg = dict(
        type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

    test_dataset = ExampleDataset()
    loader = DataLoader(test_dataset, batch_size=1)
    model = ExampleModel()
    optimizer = build_optimizer(model, optimizer_cfg)

    data_loader = DataLoader(test_dataset, batch_size=1)
    eval_hook = EpochEvalHook(data_loader, save_best=None)
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = get_logger('test_eval')
        runner = EpochBasedRunner(
            model=model,
            batch_processor=None,
            optimizer=optimizer,
            work_dir=tmpdir,
            logger=logger)
        runner.register_hook(eval_hook)
        runner.run([loader], [('train', 1)], 1)
        test_dataset.evaluate.assert_called_with(
            test_dataset, [torch.tensor([1])], logger=runner.logger)
        assert runner.meta is None or 'best_score' not in runner.meta[
            'hook_msgs']
        assert runner.meta is None or 'best_ckpt' not in runner.meta[
            'hook_msgs']

        # when `save_best` is set to 'auto', first metric will be used.
        loader = DataLoader(EvalDataset(), batch_size=1)
        model = ExampleModel()
        data_loader = DataLoader(EvalDataset(), batch_size=1)
        eval_hook = EpochEvalHook(data_loader, interval=1, save_best='auto')

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = get_logger('test_eval')
            runner = EpochBasedRunner(
                model=model,
                batch_processor=None,
                optimizer=optimizer,
                work_dir=tmpdir,
                logger=logger)
            runner.register_checkpoint_hook(dict(interval=1))
            runner.register_hook(eval_hook)
            runner.run([loader], [('train', 1)], 8)

            real_path = osp.join(tmpdir, 'epoch_4.pth')
            link_path = osp.join(tmpdir, 'best_acc.pth')

            assert runner.meta['hook_msgs']['best_ckpt'] == osp.realpath(
                real_path)
            assert osp.exists(link_path)
            assert runner.meta['hook_msgs']['best_score'] == 7

        loader = DataLoader(EvalDataset(), batch_size=1)
        model = ExampleModel()
        data_loader = DataLoader(EvalDataset(), batch_size=1)
        eval_hook = EpochEvalHook(data_loader, interval=1, save_best='acc')

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = get_logger('test_eval')
            runner = EpochBasedRunner(
                model=model,
                batch_processor=None,
                optimizer=optimizer,
                work_dir=tmpdir,
                logger=logger)
            runner.register_checkpoint_hook(dict(interval=1))
            runner.register_hook(eval_hook)
            runner.run([loader], [('train', 1)], 8)

            real_path = osp.join(tmpdir, 'epoch_4.pth')
            link_path = osp.join(tmpdir, 'best_acc.pth')

            assert runner.meta['hook_msgs']['best_ckpt'] == osp.realpath(
                real_path)
            assert osp.exists(link_path)
            assert runner.meta['hook_msgs']['best_score'] == 7

        data_loader = DataLoader(EvalDataset(), batch_size=1)
        eval_hook = EpochEvalHook(
            data_loader, interval=1, save_best='score', rule='greater')
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = get_logger('test_eval')
            runner = EpochBasedRunner(
                model=model,
                batch_processor=None,
                optimizer=optimizer,
                work_dir=tmpdir,
                logger=logger)
            runner.register_checkpoint_hook(dict(interval=1))
            runner.register_hook(eval_hook)
            runner.run([loader], [('train', 1)], 8)

            real_path = osp.join(tmpdir, 'epoch_4.pth')
            link_path = osp.join(tmpdir, 'best_score.pth')

            assert runner.meta['hook_msgs']['best_ckpt'] == osp.realpath(
                real_path)
            assert osp.exists(link_path)
            assert runner.meta['hook_msgs']['best_score'] == 7

        data_loader = DataLoader(EvalDataset(), batch_size=1)
        eval_hook = EpochEvalHook(data_loader, save_best='acc', rule='less')
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = get_logger('test_eval')
            runner = EpochBasedRunner(
                model=model,
                batch_processor=None,
                optimizer=optimizer,
                work_dir=tmpdir,
                logger=logger)
            runner.register_checkpoint_hook(dict(interval=1))
            runner.register_hook(eval_hook)
            runner.run([loader], [('train', 1)], 8)

            real_path = osp.join(tmpdir, 'epoch_6.pth')
            link_path = osp.join(tmpdir, 'best_acc.pth')

            assert runner.meta['hook_msgs']['best_ckpt'] == osp.realpath(
                real_path)
            assert osp.exists(link_path)
            assert runner.meta['hook_msgs']['best_score'] == -3

        data_loader = DataLoader(EvalDataset(), batch_size=1)
        eval_hook = EpochEvalHook(data_loader, save_best='acc')
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = get_logger('test_eval')
            runner = EpochBasedRunner(
                model=model,
                batch_processor=None,
                optimizer=optimizer,
                work_dir=tmpdir,
                logger=logger)
            runner.register_checkpoint_hook(dict(interval=1))
            runner.register_hook(eval_hook)
            runner.run([loader], [('train', 1)], 2)

            real_path = osp.join(tmpdir, 'epoch_2.pth')
            link_path = osp.join(tmpdir, 'best_acc.pth')

            assert runner.meta['hook_msgs']['best_ckpt'] == osp.realpath(
                real_path)
            assert osp.exists(link_path)
            assert runner.meta['hook_msgs']['best_score'] == 4

            resume_from = osp.join(tmpdir, 'latest.pth')
            loader = DataLoader(ExampleDataset(), batch_size=1)
            eval_hook = EpochEvalHook(data_loader, save_best='acc')
            runner = EpochBasedRunner(
                model=model,
                batch_processor=None,
                optimizer=optimizer,
                work_dir=tmpdir,
                logger=logger)
            runner.register_checkpoint_hook(dict(interval=1))
            runner.register_hook(eval_hook)
            runner.resume(resume_from)
            runner.run([loader], [('train', 1)], 8)

            real_path = osp.join(tmpdir, 'epoch_4.pth')
            link_path = osp.join(tmpdir, 'best_acc.pth')

            assert runner.meta['hook_msgs']['best_ckpt'] == osp.realpath(
                real_path)
            assert osp.exists(link_path)
            assert runner.meta['hook_msgs']['best_score'] == 7


@patch('mmaction.apis.single_gpu_test', MagicMock)
@patch('mmaction.apis.multi_gpu_test', MagicMock)
@pytest.mark.parametrize('EpochEvalHookParam',
                         (EpochEvalHook, DistEpochEvalHook))
def test_start_param(EpochEvalHookParam):
    # create dummy data
    dataloader = DataLoader(torch.ones((5, 2)))

    # 0.1. dataloader is not a DataLoader object
    with pytest.raises(TypeError):
        EpochEvalHookParam(dataloader=MagicMock(), interval=-1)

    # 0.2. negative interval
    with pytest.raises(ValueError):
        EpochEvalHookParam(dataloader, interval=-1)

    # 1. start=None, interval=1: perform evaluation after each epoch.
    runner = _build_demo_runner()
    evalhook = EpochEvalHookParam(dataloader, interval=1)
    evalhook.evaluate = MagicMock()
    runner.register_hook(evalhook)
    runner.run([dataloader], [('train', 1)], 2)
    assert evalhook.evaluate.call_count == 2  # after epoch 1 & 2

    # 2. start=1, interval=1: perform evaluation after each epoch.
    runner = _build_demo_runner()
    evalhook = EpochEvalHookParam(dataloader, start=1, interval=1)
    evalhook.evaluate = MagicMock()
    runner.register_hook(evalhook)
    runner.run([dataloader], [('train', 1)], 2)
    assert evalhook.evaluate.call_count == 2  # after epoch 1 & 2

    # 3. start=None, interval=2: perform evaluation after epoch 2, 4, 6, etc
    runner = _build_demo_runner()
    evalhook = EpochEvalHookParam(dataloader, interval=2)
    evalhook.evaluate = MagicMock()
    runner.register_hook(evalhook)
    runner.run([dataloader], [('train', 1)], 2)
    assert evalhook.evaluate.call_count == 1  # after epoch 2

    # 4. start=1, interval=2: perform evaluation after epoch 1, 3, 5, etc
    runner = _build_demo_runner()
    evalhook = EpochEvalHookParam(dataloader, start=1, interval=2)
    evalhook.evaluate = MagicMock()
    runner.register_hook(evalhook)
    runner.run([dataloader], [('train', 1)], 3)
    assert evalhook.evaluate.call_count == 2  # after epoch 1 & 3

    # 5. start=0/negative, interval=1: perform evaluation after each epoch and
    #    before epoch 1.
    runner = _build_demo_runner()
    evalhook = EpochEvalHookParam(dataloader, start=0)
    evalhook.evaluate = MagicMock()
    runner.register_hook(evalhook)
    runner.run([dataloader], [('train', 1)], 2)
    assert evalhook.evaluate.call_count == 3  # before epoch1 and after e1 & e2

    runner = _build_demo_runner()
    with pytest.warns(UserWarning):
        evalhook = EpochEvalHookParam(dataloader, start=-2)
    evalhook.evaluate = MagicMock()
    runner.register_hook(evalhook)
    runner.run([dataloader], [('train', 1)], 2)
    assert evalhook.evaluate.call_count == 3  # before epoch1 and after e1 & e2

    # 6. resuming from epoch i, start = x (x<=i), interval =1: perform
    #    evaluation after each epoch and before the first epoch.
    runner = _build_demo_runner()
    evalhook = EpochEvalHookParam(dataloader, start=1)
    evalhook.evaluate = MagicMock()
    runner.register_hook(evalhook)
    runner._epoch = 2
    runner.run([dataloader], [('train', 1)], 3)
    assert evalhook.evaluate.call_count == 2  # before & after epoch 3

    # 7. resuming from epoch i, start = i+1/None, interval =1: perform
    #    evaluation after each epoch.
    runner = _build_demo_runner()
    evalhook = EpochEvalHookParam(dataloader, start=2)
    evalhook.evaluate = MagicMock()
    runner.register_hook(evalhook)
    runner._epoch = 1
    runner.run([dataloader], [('train', 1)], 3)
    assert evalhook.evaluate.call_count == 2  # after epoch 2 & 3
