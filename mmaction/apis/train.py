import copy as cp

import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (DistSamplerSeedHook, EpochBasedRunner, OptimizerHook,
                         build_optimizer)
from mmcv.runner.hooks import Fp16OptimizerHook

from ..core import (DistEpochEvalHook, EpochEvalHook,
                    OmniSourceDistSamplerSeedHook, OmniSourceRunner)
from ..datasets import build_dataloader, build_dataset
from ..utils import get_root_logger


def train_model(model,
                dataset,
                cfg,
                distributed=False,
                validate=False,
                timestamp=None,
                meta=None):
    """Train model entry function.

    Args:
        model (nn.Module): The model to be trained.
        dataset (:obj:`Dataset`): Train dataset.
        cfg (dict): The config dict for training.
        distributed (bool): Whether to use distributed training.
            Default: False.
        validate (bool): Whether to do evaluation. Default: False.
        timestamp (str | None): Local time for runner. Default: None.
        meta (dict | None): Meta dict to record some important information.
            Default: None
    """
    logger = get_root_logger(log_level=cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    if cfg.omnisource:
        # The option can override videos_per_gpu
        train_ratio = cfg.data.get('train_ratio', None)
        omni_videos_per_gpu = cfg.data.get('omni_videos_per_gpu', None)
        dataloader_setting_tmpl = dict(
            videos_per_gpu=cfg.data.get('videos_per_gpu', {}),
            workers_per_gpu=cfg.data.get('workers_per_gpu', {}),
            num_gpus=len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed)
        if omni_videos_per_gpu is None:
            dataloader_setting = dict(dataloader_setting_tmpl,
                                      **cfg.data.get('train_dataloader', {}))
            data_loaders = [
                build_dataloader(ds, **dataloader_setting) for ds in dataset
            ]
        else:
            dataloader_settings = []
            for videos_per_gpu in omni_videos_per_gpu:
                dataloader_setting = cp.deepcopy(dataloader_setting_tmpl)
                dataloader_setting['videos_per_gpu'] = videos_per_gpu
                dataloader_settings.append(dataloader_setting)
            data_loaders = [
                build_dataloader(ds, **setting)
                for ds, setting in zip(dataset, dataloader_settings)
            ]

    else:
        dataloader_setting = dict(
            videos_per_gpu=cfg.data.get('videos_per_gpu', 2),
            workers_per_gpu=cfg.data.get('workers_per_gpu', 0),
            # cfg.gpus will be ignored if distributed
            num_gpus=len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed)
        dataloader_setting = dict(dataloader_setting,
                                  **cfg.data.get('train_dataloader', {}))

        data_loaders = [
            build_dataloader(ds, **dataloader_setting) for ds in dataset
        ]

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    if cfg.omnisource:
        runner = OmniSourceRunner(
            model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta)
    else:
        runner = EpochBasedRunner(
            model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta)
    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))
    if distributed:
        if cfg.omnisource:
            runner.register_hook(OmniSourceDistSamplerSeedHook())
        else:
            runner.register_hook(DistSamplerSeedHook())

    if validate:
        eval_cfg = cfg.get('evaluation', {})
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        dataloader_setting = dict(
            videos_per_gpu=cfg.data.get('videos_per_gpu', 2),
            workers_per_gpu=cfg.data.get('workers_per_gpu', 0),
            # cfg.gpus will be ignored if distributed
            num_gpus=len(cfg.gpu_ids),
            dist=distributed,
            shuffle=False)
        dataloader_setting = dict(dataloader_setting,
                                  **cfg.data.get('val_dataloader', {}))
        val_dataloader = build_dataloader(val_dataset, **dataloader_setting)
        eval_hook = DistEpochEvalHook if distributed else EpochEvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    # All hooks are registered now, for omnisource experiments, we need to
    # override the `end_of_epoch` method for each hook.
    # if cfg.omnisource:
    #
    #     def new_end_of_epoch(runner):
    #         return runner.inner_iter + 1 == len(runner.main_loader)
    #
    #     for hook in runner._hooks:
    #         hook.end_of_epoch = new_end_of_epoch

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    if cfg.omnisource:
        runner.run(
            data_loaders,
            cfg.workflow,
            cfg.total_epochs,
            train_ratio=train_ratio)
    else:
        runner.run(data_loaders, cfg.workflow, cfg.total_epochs)
