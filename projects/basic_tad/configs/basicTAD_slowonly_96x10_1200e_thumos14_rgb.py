# model settings
model = dict(type='mmdet.SingleStageDetector',
             backbone=dict(type='SlowOnly'),
             neck=[
                 dict(
                     type='VDM',
                     in_channels=2048,
                     out_channels=512,
                     conv_cfg=dict(type='Conv3d'),
                     norm_cfg=dict(type='SyncBN'),
                     kernel_sizes=(3, 1, 1),
                     strides=(2, 1, 1),
                     paddings=(1, 0, 0),
                     stage_layers=(1, 1, 1, 1),
                     out_indices=(0, 1, 2, 3, 4),
                     out_pooling=True),
                 dict(type='mmdet.FPN',
                      in_channels=[2048, 512, 512, 512, 512],
                      out_channels=256,
                      num_outs=5,
                      conv_cfg=dict(type='Conv1d'),
                      norm_cfg=dict(type='SyncBN'))],
             bbox_head=dict(
                 type='RetinaHead1D',
                 num_classes=20,
                 in_channels=256,
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='SyncBN'),
                 anchor_generator=dict(
                     type='mmdet.AnchorGenerator',
                     octave_base_scale=2,
                     scales_per_octave=5,
                     ratios=[1.0],
                     strides=[1, 2, 4, 8, 16]),
                 bbox_coder=dict(
                     type='mmdet.DeltaXYWHBBoxCoder',
                     target_means=[.0, .0, .0, .0],
                     target_stds=[1.0, 1.0, 1.0, 1.0]),
                 reg_decoded_bbox=True,
                 loss_cls=dict(type='mmdet.FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
                 loss_bbox=dict(type='DIoU1DLoss', loss_weight=1.0),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv1d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='retina_cls',
                         std=0.01,
                         bias_prob=0.01))),
             data_preprocessor=dict(
                 type='ActionDataPreprocessor',
                 mean=[123.675, 116.28, 103.53],
                 std=[58.395, 57.12, 57.375],
                 format_shape='NCTHW'),
             train_cfg=dict(
                 assigner=dict(
                     type='mmdet.MaxIoUAssigner',
                     pos_iou_thr=0.6,
                     neg_iou_thr=0.4,
                     min_pos_iou=0,
                     ignore_iof_thr=-1,
                     ignore_wrt_candidates=True,
                     iou_calculator=dict(type='BboxOverlaps1D')),
                 allowed_border=-1,
                 pos_weight=-1,
                 debug=False),
             test_cfg=dict(nms_pre=300, score_thr=0.005))  # we perform NMS in Metric rather than in the model

# dataset settings
data_root = 'data/thumos14'  # Root path to data for training
data_prefix_train = 'rawframes/val'  # path to data for training
data_prefix_val = 'rawframes/test'  # path to data for validation and testing
ann_file_train = 'annotations/basicTAD/val.json'  # Path to the annotation file for training
ann_file_val = 'annotations/basicTAD/test.json'  # Path to the annotation file for validation
ann_file_test = ann_file_val

clip_len = 96
frame_interval = 10
img_shape = (112, 112)
img_shape_test = (128, 128)
overlap_ratio = 0.25

train_pipeline = [
    dict(type='Time2Frame'),
    dict(type='TemporalRandomCrop',
         clip_len=clip_len,
         frame_interval=frame_interval,
         iof_th=0.75),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(128, -1), keep_ratio=True),  # scale images' short-side to 128, keep aspect ratio
    dict(type='SpatialRandomCrop', crop_size=img_shape),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='PhotoMetricDistortion',
         brightness_delta=32,
         contrast_range=(0.5, 1.5),
         saturation_range=(0.5, 1.5),
         hue_delta=18,
         p=0.5),
    dict(type='Rotate',
         limit=(-45, 45),
         border_mode='reflect_101',
         p=0.5),
    dict(type='Pad', size=(clip_len, *img_shape)),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackTadInputs',
         meta_keys=('img_id', 'img_shape', 'pad_shape', 'scale_factor',))
]
val_pipeline = [
    dict(type='Time2Frame'),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(128, -1), keep_ratio=True),
    dict(type='SpatialCenterCrop', crop_size=img_shape_test),
    dict(type='Pad', size=(clip_len, *img_shape_test)),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackTadInputs',
         meta_keys=('img_id', 'img_shape', 'scale_factor', 'offset_sec'))
]
# test_pipeline = val_pipeline

train_dataloader = dict(  # Config of train dataloader
    batch_size=2,  # Batch size of each single GPU during training
    num_workers=6,  # Workers to pre-fetch data for each single GPU during training
    persistent_workers=True,
    # If `True`, the dataloader will not shut down the worker processes after an epoch end, which can accelerate training speed
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(  # Config of train dataset
        type='Thumos14Dataset',
        filename_tmpl='img_{:05}.jpg',
        ann_file=ann_file_train,  # Path of annotation file
        data_root=data_root,  # Root path to data, including both frames and ann_file
        data_prefix=dict(imgs=data_prefix_train),  # Prefix of specific data, e.g., frames and ann_file
        pipeline=train_pipeline))
val_dataloader = dict(  # Config of validation dataloader
    batch_size=2,  # Batch size of each single GPU during validation
    num_workers=6,  # Workers to pre-fetch data for each single GPU during validation
    persistent_workers=True,  # If `True`, the dataloader will not shut down the worker processes after an epoch end
    sampler=dict(type='DefaultSampler', shuffle=False),  # Not shuffle during validation and testing
    # DefaultSampler which supports both distributed and non-distributed training. Refer to https://github.com/open-mmlab/mmengine/blob/main/mmengine/dataset/sampler.py)  # Randomly shuffle the training data in each epoch
    dataset=dict(  # Config of validation dataset
        type='Thumos14ValDataset',
        clip_len=clip_len, frame_interval=frame_interval, overlap_ratio=overlap_ratio,
        filename_tmpl='img_{:05}.jpg',
        ann_file=ann_file_val,  # Path of annotation file
        data_root=data_root,
        data_prefix=dict(imgs=data_prefix_val),  # Prefix of specific data components
        pipeline=val_pipeline,
        test_mode=True))
test_dataloader = val_dataloader

# evaluation settings
val_evaluator = dict(  # My customized evaluator for mean average precision
    type='TADmAPMetric',
    metric='mAP',
    iou_thrs=[0.3, 0.4, 0.5, 0.6, 0.7],
    nms_cfg=dict(type='nms', iou_thr=0.6))
test_evaluator = val_evaluator  # Config of testing evaluator

train_cfg = dict(  # Config of training loop
    type='EpochBasedTrainLoop',  # Name of training loop
    max_epochs=1200,  # Total training epochs
    val_begin=1,  # The epoch that begins validating
    val_interval=100)  # Validation interval
val_cfg = dict(  # Config of validation loop
    type='ValLoop')  # Name of validation loop
test_cfg = dict(  # Config of testing loop
    type='TestLoop')  # Name of testing loop

# learning policy
param_scheduler = [  # Parameter scheduler for updating optimizer parameters, support dict or list
    # Linear learning rate warm-up scheduler
    dict(type='LinearLR',
         start_factor=0.1,
         by_epoch=True,
         begin=0,
         end=40,
         convert_to_iter_based=True),
    dict(type='CosineRestartLR',  # Decays the learning rate once the number of epoch reaches one of the milestones
         periods=[100] * 12,
         restart_weights=[1] * 12,
         eta_min=1e-4,  # The min_lr, note it's NOT the min_lr_ratio
         by_epoch=True,
         begin=40,
         end=1240,
         convert_to_iter_based=True)]  # Convert to update by iteration.

# optimizer
optim_wrapper = dict(  # Config of optimizer wrapper
    type='OptimWrapper',  # Name of optimizer wrapper, switch to AmpOptimWrapper to enable mixed precision training
    optimizer=dict(
        # Config of optimizer. Support all kinds of optimizers in PyTorch. Refer to https://pytorch.org/docs/stable/optim.html#algorithms
        type='SGD',  # Name of optimizer
        lr=0.01,  # Learning rate
        momentum=0.9,  # Momentum factor
        weight_decay=0.0001),  # Weight decay
    clip_grad=dict(max_norm=40, norm_type=2))  # Config of gradient clip
auto_scale_lr = dict(enable=False, base_batch_size=16)  # The lr=0.01 is for batch_size=16.

# runtime settings
# imports
custom_imports = dict(imports=['models'], allow_failed_imports=False)
default_scope = 'mmaction'  # The default registry scope to find modules. Refer to https://mmengine.readthedocs.io/en/latest/tutorials/registry.html
default_hooks = dict(  # Hooks to execute default actions like updating model parameters and saving checkpoints.
    runtime_info=dict(type='RuntimeInfoHook'),  # The hook to updates runtime information into message hub
    timer=dict(type='IterTimerHook'),  # The logger used to record time spent during iteration
    logger=dict(
        type='LoggerHook',  # The logger used to record logs during training/validation/testing phase
        interval=20,  # Interval to print the log
        ignore_last=False,
        interval_exp_name=1000),  # Ignore the log of last iterations in each epoch
    param_scheduler=dict(type='ParamSchedulerHook'),  # The hook to update some hyper-parameters in optimizer
    checkpoint=dict(
        type='CheckpointHook',  # The hook to save checkpoints periodically
        interval=100,  # The saving period
        save_best='auto',  # Specified metric to mearsure the best checkpoint during evaluation
        max_keep_ckpts=10),  # The maximum checkpoints to keep
    sampler_seed=dict(type='DistSamplerSeedHook'),  # Data-loading sampler for distributed training
    sync_buffers=dict(type='SyncBuffersHook'))  # Synchronize model buffers at the end of each epoch
env_cfg = dict(  # Dict for setting environment
    cudnn_benchmark=False,  # Whether to enable cudnn benchmark
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),  # Parameters to setup multiprocessing
    dist_cfg=dict(backend='nccl'))  # Parameters to setup distributed environment, the port can also be set

log_processor = dict(
    type='LogProcessor',  # Log processor used to format log information
    window_size=20,  # Default smooth interval
    by_epoch=True)  # Whether to format logs with epoch type
vis_backends = [  # List of visualization backends
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')]  # Local visualization backend
visualizer = dict(  # Config of visualizer
    type='ActionVisualizer',  # Name of visualizer
    vis_backends=vis_backends)
# randomness = dict(seed=10, deterministic=True)
# find_unused_parameters = True
log_level = 'INFO'  # The level of logging
load_from = None  # Load model checkpoint as a pre-trained model from a given path. This will not resume training.
resume = False  # Whether to resume from the checkpoint defined in `load_from`. If `load_from` is None, it will resume the latest checkpoint in the `work_dir`.
