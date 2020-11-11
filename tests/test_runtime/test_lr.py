import logging
import shutil
import sys
import tempfile
from unittest.mock import MagicMock, call

import torch
import torch.nn as nn
from mmcv.runner import IterTimerHook, PaviLoggerHook, build_runner
from torch.utils.data import DataLoader


def test_tin_lr_updater_hook():
    sys.modules['pavi'] = MagicMock()
    loader = DataLoader(torch.ones((10, 2)))
    runner = _build_demo_runner()

    hook_cfg = dict(type='TINLrUpdaterHook', min_lr=0.1)
    runner.register_hook_from_cfg(hook_cfg)

    hook_cfg = dict(
        type='TINLrUpdaterHook',
        by_epoch=False,
        min_lr=0.1,
        warmup='exp',
        warmup_iters=2,
        warmup_ratio=0.9)
    runner.register_hook_from_cfg(hook_cfg)
    runner.register_hook_from_cfg(dict(type='IterTimerHook'))
    runner.register_hook(IterTimerHook())

    hook_cfg = dict(
        type='TINLrUpdaterHook',
        by_epoch=False,
        min_lr=0.1,
        warmup='constant',
        warmup_iters=2,
        warmup_ratio=0.9)
    runner.register_hook_from_cfg(hook_cfg)
    runner.register_hook_from_cfg(dict(type='IterTimerHook'))
    runner.register_hook(IterTimerHook())

    hook_cfg = dict(
        type='TINLrUpdaterHook',
        by_epoch=False,
        min_lr=0.1,
        warmup='linear',
        warmup_iters=2,
        warmup_ratio=0.9)
    runner.register_hook_from_cfg(hook_cfg)
    runner.register_hook_from_cfg(dict(type='IterTimerHook'))
    runner.register_hook(IterTimerHook())
    # add pavi hook
    hook = PaviLoggerHook(interval=1, add_graph=False, add_last_ckpt=True)
    runner.register_hook(hook)
    runner.run([loader], [('train', 1)])
    shutil.rmtree(runner.work_dir)

    assert hasattr(hook, 'writer')
    calls = [
        call('train', {
            'learning_rate': 0.028544155877284292,
            'momentum': 0.95
        }, 1),
        call('train', {
            'learning_rate': 0.04469266270539641,
            'momentum': 0.95
        }, 6),
        call('train', {
            'learning_rate': 0.09695518130045147,
            'momentum': 0.95
        }, 10)
    ]
    hook.writer.add_scalars.assert_has_calls(calls, any_order=True)


def _build_demo_runner(runner_type='EpochBasedRunner',
                       max_epochs=1,
                       max_iters=None):

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

    optimizer = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.95)

    log_config = dict(
        interval=1, hooks=[
            dict(type='TextLoggerHook'),
        ])

    tmp_dir = tempfile.mkdtemp()
    runner = build_runner(
        dict(type=runner_type),
        default_args=dict(
            model=model,
            work_dir=tmp_dir,
            optimizer=optimizer,
            logger=logging.getLogger(),
            max_epochs=max_epochs,
            max_iters=max_iters))
    runner.register_checkpoint_hook(dict(interval=1))
    runner.register_logger_hooks(log_config)
    return runner
