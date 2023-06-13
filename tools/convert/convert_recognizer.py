# Copyright (c) OpenMMLab. All rights reserved.
import argparse

from mmengine.config import Config


def convert(config_path, output_config_path):
    print('start convert')

    cfg = Config.fromfile(config_path)
    origin_dataset_type = cfg.dataset_type

    # dataset
    if origin_dataset_type != 'VideoDataset':
        cfg.dataset_type = 'VideoDataset'
        cfg.data_root = 'data/kinetics400/rawframes_train'
        cfg.data_root_val = 'data/kinetics400/rawframes_val'
        cfg.ann_file_train = \
            'data/kinetics400/kinetics400_train_list_rawframes.txt'
        cfg.ann_file_val = \
            'data/kinetics400/kinetics400_val_list_rawframes.txt'
        cfg.ann_file_test = \
            'data/kinetics400/kinetics400_val_list_rawframes.txt'

    # model
    preprocess_cfg = cfg.img_norm_cfg

    formatshape = None
    for trans in cfg.train_pipeline:
        if trans.type == 'FormatShape':
            formatshape = trans.input_format

    preprocess_cfg['input_format'] = formatshape
    cfg.preprocess_cfg = preprocess_cfg

    cfg.model.data_preprocessor = dict(
        type='ActionDataPreprocessor', **dict(preprocess_cfg))
    cfg.pop('img_norm_cfg')

    if (cfg.model.test_cfg is not None) and ('average_clips'
                                             in cfg.model.test_cfg):
        cfg.model.cls_head.average_clips = cfg.model.test_cfg.average_clips
        cfg.model.test_cfg.pop('average_clips')
        if len(cfg.model.test_cfg) == 0:
            cfg.model.test_cfg = None

    # pipeline
    pipelines = [cfg.train_pipeline, cfg.val_pipeline, cfg.test_pipeline]

    if origin_dataset_type == 'VideoDataset':

        for pipeline in pipelines:
            new_pipeline = [
                trans for trans in pipeline
                if trans.type not in ['Normalize', 'Collect', 'ToTensor']
            ]
            new_pipeline.append(dict(type='PackActionInputs'))
            pipeline.clear().extend(new_pipeline)

    elif origin_dataset_type == 'RawframeDataset':

        for pipeline in pipelines:
            new_pipeline = [
                trans for trans in pipeline if trans.type not in
                ['RawFrameDecode', 'Normalize', 'Collect', 'ToTensor']
            ]
            new_pipeline.insert(0, dict(type='DecordInit'))
            new_pipeline.insert(2, dict(type='DecordDecode'))
            new_pipeline.append(dict(type='PackActionInputs'))
            pipeline.clear()
            pipeline.extend(new_pipeline)

    # dataloader
    cfg.data.train.update(
        dict(
            type=cfg.dataset_type,
            ann_file=cfg.ann_file_train,
            data_prefix=dict(video=cfg.data_root),
            pipeline=cfg.train_pipeline))
    cfg.train_dataloader = dict(
        batch_size=cfg.data.videos_per_gpu,
        num_workers=cfg.data.workers_per_gpu,
        persistent_workers=True,
        sampler=dict(type='DefaultSampler', shuffle=True),
        dataset=cfg.data.train)

    val_batchsize = cfg.data.val_dataloader.videos_per_gpu \
        if 'val_dataloader' in cfg.data else cfg.data.videos_per_gpu
    cfg.data.val.update(
        dict(
            type=cfg.dataset_type,
            ann_file=cfg.ann_file_val,
            data_prefix=dict(video=cfg.data_root_val),
            pipeline=cfg.val_pipeline))
    cfg.val_dataloader = dict(
        batch_size=val_batchsize,
        num_workers=cfg.data.workers_per_gpu,
        persistent_workers=True,
        sampler=dict(type='DefaultSampler', shuffle=True),
        dataset=cfg.data.val)

    test_batchsize = cfg.data.test_dataloader.videos_per_gpu \
        if 'test_dataloader' in cfg.data else cfg.data.videos_per_gpu

    cfg.data.test.update(
        dict(
            type=cfg.dataset_type,
            ann_file=cfg.ann_file_test,
            data_prefix=dict(video=cfg.data_root_val),
            pipeline=cfg.test_pipeline))
    cfg.test_dataloader = dict(
        batch_size=test_batchsize,
        num_workers=cfg.data.workers_per_gpu,
        persistent_workers=True,
        sampler=dict(type='DefaultSampler', shuffle=True),
        dataset=cfg.data.test)

    cfg.pop('data')

    # eval
    cfg.val_evaluator = dict(type='AccMetric')
    cfg.test_evaluator = cfg.val_evaluator

    cfg.val_cfg = dict(interval=cfg.evaluation.interval)
    cfg.test_cfg = dict()
    cfg.pop('evaluation')
    # optimizer
    optimizer_wrapper = dict(optimizer=dict())
    for k, v in cfg.optimizer.items():
        if k not in ['paramwise_cfg', 'constructor']:
            optimizer_wrapper['optimizer'].update({k: v})
        else:
            optimizer_wrapper.update({k: v})
    for k, v in cfg.optimizer_config.items():
        if k == 'grad_clip':
            k = 'clip_grad'
        optimizer_wrapper.update({k: v})
    cfg.optimizer_wrapper = optimizer_wrapper

    cfg.pop('optimizer')
    cfg.pop('optimizer_config')
    # train_cfg
    cfg.train_cfg = dict(by_epoch=True, max_epochs=cfg.total_epochs)
    cfg.pop('total_epochs')
    # schedule
    cfg.param_scheduler = []
    warmup_epoch = 0
    if 'warmup' in cfg.lr_config:
        warmup_ratio = 0.1 \
            if 'warmup_ratio' not in cfg.lr_config \
            else cfg.lr_config.warmup_ratio

        warmup_epoch = cfg.lr_config.warmup_iters
        cfg.param_scheduler.append(
            dict(
                type='LinearLR',
                bengin=0,
                start_factor=warmup_ratio,
                end=cfg.lr_config.warmup_iters,
                by_epoch=cfg.lr_config.warmup_by_epoch))
    if cfg.lr_config.policy == 'step':
        cfg.param_scheduler.append(
            dict(
                type='MultiStepLR',
                milestones=cfg.lr_config.step,
                by_epoch=cfg.train_cfg.by_epoch,
                begin=0,
                end=cfg.train_cfg.max_epochs,
                gamma=0.1
                if 'gamma' not in cfg.lr_config else cfg.lr_config.gamma))
    elif cfg.lr_config.policy == 'CosineAnnealing':
        cfg.param_scheduler.append(
            dict(
                type='CosineAnnealingLR',
                eta_min=cfg.lr_config.min_lr,
                by_epoch=cfg.train_cfg.by_epoch,
                begin=warmup_epoch,
                end=cfg.train_cfg.max_epochs,
                T_max=cfg.train_cfg.max_epochs - warmup_epoch))
    else:
        raise ValueError(f'Not support convert {cfg.lr_config.policy}')
    cfg.pop('lr_config')

    # runtime
    cfg.default_scope = 'mmaction'

    cfg.default_hooks = dict(
        runtime_info=dict(type='RuntimeInfoHook'),
        timer=dict(type='IterTimerHook'),
        logger=dict(type='LoggerHook', interval=20),
        param_scheduler=dict(type='ParamSchedulerHook'),
        checkpoint=dict(type='CheckpointHook', **cfg.checkpoint_config),
        sampler_seed=dict(type='DistSamplerSeedHook'),
    )

    cfg.env_cfg = dict(
        cudnn_benchmark=False,
        mp_cfg=dict(
            mp_start_method=cfg.mp_start_method,
            opencv_num_threads=cfg.opencv_num_threads),
        dist_cfg=dict(**cfg.dist_params),
    )

    cfg.log_level = 'INFO'
    cfg.load_from = None
    cfg.resume = False

    cfg.pop('workflow')
    cfg.pop('mp_start_method')
    cfg.pop('opencv_num_threads')
    cfg.pop('log_config')
    cfg.pop('dist_params')
    cfg.pop('checkpoint_config')
    cfg.pop('work_dir')
    cfg.dump(output_config_path)

    print('Successful')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert an action recognizer config file \
        from OpenMMLAb framework v1.0 to v2.0')
    parser.add_argument('config', help='The config file path')
    parser.add_argument('output_config', help='The config file path')
    args = parser.parse_args()
    convert(args.config, args.output_config)
