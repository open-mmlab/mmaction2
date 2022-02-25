# Copyright (c) OpenMMLab. All rights reserved.
import copy
import tempfile
from collections import OrderedDict

import pytest
import torch
import torch.nn as nn
from mmcv import Config
from torch.utils.data import Dataset

from mmaction.apis import train_model
from mmaction.datasets import DATASETS


@DATASETS.register_module()
class ExampleDataset(Dataset):

    def __init__(self, test_mode=False):
        self.test_mode = test_mode

    @staticmethod
    def evaluate(results, logger=None):
        eval_results = OrderedDict()
        eval_results['acc'] = 1
        return eval_results

    def __getitem__(self, idx):
        results = dict(imgs=torch.tensor([1]))
        return results

    def __len__(self):
        return 1


class ExampleModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.test_cfg = None
        self.conv1 = nn.Conv2d(3, 8, kernel_size=1)
        self.norm1 = nn.BatchNorm1d(2)

    def forward(self, imgs, return_loss=False):
        self.norm1(torch.rand(3, 2).cuda())
        losses = dict()
        losses['test_loss'] = torch.tensor([0.5], requires_grad=True)
        return losses

    def train_step(self, data_batch, optimizer, **kwargs):
        imgs = data_batch['imgs']
        losses = self.forward(imgs, True)
        loss = torch.tensor([0.5], requires_grad=True)
        outputs = dict(loss=loss, log_vars=losses, num_samples=3)
        return outputs

    def val_step(self, data_batch, optimizer, **kwargs):
        imgs = data_batch['imgs']
        self.forward(imgs, False)
        outputs = dict(results=0.5)
        return outputs


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires CUDA support')
def test_train_model():
    model = ExampleModel()
    dataset = ExampleDataset()
    datasets = [ExampleDataset(), ExampleDataset()]
    _cfg = dict(
        seed=0,
        gpus=1,
        gpu_ids=[0],
        resume_from=None,
        load_from=None,
        workflow=[('train', 1)],
        total_epochs=5,
        evaluation=dict(interval=1, save_best='acc'),
        data=dict(
            videos_per_gpu=1,
            workers_per_gpu=0,
            val=dict(type='ExampleDataset')),
        optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001),
        optimizer_config=dict(grad_clip=dict(max_norm=40, norm_type=2)),
        lr_config=dict(policy='step', step=[40, 80]),
        omnisource=False,
        precise_bn=False,
        checkpoint_config=dict(interval=1),
        log_level='INFO',
        log_config=dict(interval=20, hooks=[dict(type='TextLoggerHook')]))

    with tempfile.TemporaryDirectory() as tmpdir:
        # normal train
        cfg = copy.deepcopy(_cfg)
        cfg['work_dir'] = tmpdir
        config = Config(cfg)
        train_model(model, dataset, config)

    with tempfile.TemporaryDirectory() as tmpdir:
        # train with validation
        cfg = copy.deepcopy(_cfg)
        cfg['work_dir'] = tmpdir
        config = Config(cfg)
        train_model(model, dataset, config, validate=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = copy.deepcopy(_cfg)
        cfg['work_dir'] = tmpdir
        cfg['omnisource'] = True
        config = Config(cfg)
        train_model(model, datasets, config)

    with tempfile.TemporaryDirectory() as tmpdir:
        # train with precise_bn on
        cfg = copy.deepcopy(_cfg)
        cfg['work_dir'] = tmpdir
        cfg['workflow'] = [('train', 1), ('val', 1)]
        cfg['data'] = dict(
            videos_per_gpu=1,
            workers_per_gpu=0,
            train=dict(type='ExampleDataset'),
            val=dict(type='ExampleDataset'))
        cfg['precise_bn'] = dict(num_iters=1, interval=1)
        config = Config(cfg)
        train_model(model, datasets, config)
