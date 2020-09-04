import os.path as osp
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
from mmaction.datasets import RawframeDataset


class ExampleDataset(Dataset):

    def __init__(self):
        self.index = 0
        self.eval_result = [1, 4, 3, 7, 2, -3, 4, 6]

    def __getitem__(self, idx):
        results = dict(imgs=torch.tensor([1]))
        return results

    def __len__(self):
        return 10

    @mock.create_autospec
    def evaluate(self, results, logger=None):
        pass


data_prefix = 'tests/data'
frame_ann_file = osp.join(data_prefix, 'frame_test_list.txt')

frame_pipeline = [
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1),
    dict(type='RawFrameDecode', io_backend='disk'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False)
]


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


def test_multi_grid_hook():
    with pytest.raises(AssertionError):
        # cfg must have fields 'data' and 'multi_grid'
        cfg = {'data': dict(size=224), 'unsupported': 123}
        multi_grid_hook = MultiGridHook(cfg)

    optimizer_cfg = dict(
        type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

    test_dataset = RawframeDataset(
        frame_ann_file,
        frame_pipeline,
        data_prefix,
        short_cycle_factors=[0.5, 0.5**0.5])
    loader = DataLoader(test_dataset, batch_size=1)
    model = ExampleModel()
    optimizer = build_optimizer(model, optimizer_cfg)

    # test multigrid
    cfg = mmcv.Config.fromfile(
        'configs/recognition/i3d/i3d_r50_multigrid_32x2x1_100e_kinetics400_rgb.py'  # noqa: E501
    )
    # Skip the subbn3d since it is hardcoded to use cuda
    cfg.model.backbone.norm_cfg = dict(type='BN3d')
    multi_grid_hook = MultiGridHook(cfg)
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = get_logger('test_multi_grid')
        runner = EpochBasedRunner(
            model=model,
            batch_processor=None,
            optimizer=optimizer,
            work_dir=tmpdir,
            logger=logger)
        runner.register_training_hooks(
            cfg.lr_config, log_config=cfg.log_config)
        runner.register_hook(multi_grid_hook)
        runner.run([loader], [('train', 1)], 2)

    with pytest.raises(NotImplementedError):
        # cfg must use step learning rate policy
        cfg.lr_config = dict(policy='CosineAnnealing', min_lr=0)
        multi_grid_hook = MultiGridHook(cfg)
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = get_logger('test_longcycle')
            runner = EpochBasedRunner(
                model=model,
                batch_processor=None,
                optimizer=optimizer,
                work_dir=tmpdir,
                logger=logger)
            runner.register_training_hooks(
                cfg.lr_config, log_config=cfg.log_config)
            runner.register_hook(multi_grid_hook)
            runner.run([loader], [('train', 1)], 2)

    with pytest.raises(ValueError):
        # Multigrid should have at least longcycle
        cfg = mmcv.Config.fromfile(
            'configs/recognition/i3d/i3d_r50_multigrid_32x2x1_100e_kinetics400_rgb.py'  # noqa: E501
        )
        cfg.multi_grid.long_cycle = False
        multi_grid_hook = MultiGridHook(cfg)
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = get_logger('test_longcycle')
            runner = EpochBasedRunner(
                model=model,
                batch_processor=None,
                optimizer=optimizer,
                work_dir=tmpdir,
                logger=logger)
            runner.register_training_hooks(
                cfg.lr_config, log_config=cfg.log_config)
            runner.register_hook(multi_grid_hook)
            runner.run([loader], [('train', 1)], 2)
