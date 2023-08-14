# model settings
model = dict(  # Config of the model
    type='Recognizer2D',  # Class name of the recognizer
    backbone=dict(  # Dict for backbone
        type='ResNet',  # Name of the backbone
        pretrained='torchvision://resnet50',  # The url/site of the pretrained model
        depth=50,  # Depth of ResNet model
        norm_eval=False),  # Whether to set BN layers to eval mode when training
    cls_head=dict(  # Dict for classification head
        type='TSNHead',  # Name of classification head
        num_classes=400,  # Number of classes to be classified.
        in_channels=2048,  # The input channels of classification head.
        spatial_type='avg',  # Type of pooling in spatial dimension
        consensus=dict(type='AvgConsensus', dim=1),  # Config of consensus module
        dropout_ratio=0.4,  # Probability in dropout layer
        init_std=0.01, # Std value for linear layer initiation
        average_clips='prob'),  # Method to average multiple clip results
    data_preprocessor=dict(  # Dict for data preprocessor
        type='ActionDataPreprocessor',  # Name of data preprocessor
        mean=[123.675, 116.28, 103.53],  # Mean values of different channels to normalize
        std=[58.395, 57.12, 57.375],  # Std values of different channels to normalize
        format_shape='NCHW'),  # Final image shape format
    # model training and testing settings
    train_cfg=None,  # Config of training hyperparameters for TSN
    test_cfg=None)  # Config for testing hyperparameters for TSN.

# dataset settings
dataset_type = 'RawframeDataset'  # Type of dataset for training, validation and testing
data_root = 'data/kinetics400/rawframes_train/'  # Root path to data for training
data_root_val = 'data/kinetics400/rawframes_val/'  # Root path to data for validation and testing
ann_file_train = 'data/kinetics400/kinetics400_train_list_rawframes.txt'  # Path to the annotation file for training
ann_file_val = 'data/kinetics400/kinetics400_val_list_rawframes.txt'  # Path to the annotation file for validation
ann_file_test = 'data/kinetics400/kinetics400_val_list_rawframes.txt'  # Path to the annotation file for testing

train_pipeline = [  # Training data processing pipeline
    dict(  # Config of SampleFrames
        type='SampleFrames',  # Sample frames pipeline, sampling frames from video
        clip_len=1,  # Frames of each sampled output clip
        frame_interval=1,  # Temporal interval of adjacent sampled frames
        num_clips=3),  # Number of clips to be sampled
    dict(  # Config of RawFrameDecode
        type='RawFrameDecode'),  # Load and decode Frames pipeline, picking raw frames with given indices
    dict(  # Config of Resize
        type='Resize',  # Resize pipeline
        scale=(-1, 256)),  # The scale to resize images
    dict(  # Config of MultiScaleCrop
        type='MultiScaleCrop',  # Multi scale crop pipeline, cropping images with a list of randomly selected scales
        input_size=224,  # Input size of the network
        scales=(1, 0.875, 0.75, 0.66),  # Scales of width and height to be selected
        random_crop=False,  # Whether to randomly sample cropping bbox
        max_wh_scale_gap=1),  # Maximum gap of w and h scale levels
    dict(  # Config of Resize
        type='Resize',  # Resize pipeline
        scale=(224, 224),  # The scale to resize images
        keep_ratio=False),  # Whether to resize with changing the aspect ratio
    dict(  # Config of Flip
        type='Flip',  # Flip Pipeline
        flip_ratio=0.5),  # Probability of implementing flip
    dict(  # Config of FormatShape
        type='FormatShape',  # Format shape pipeline, Format final image shape to the given input_format
        input_format='NCHW'),  # Final image shape format
    dict(type='PackActionInputs')  # Config of PackActionInputs
]
val_pipeline = [  # Validation data processing pipeline
    dict(  # Config of SampleFrames
        type='SampleFrames',  # Sample frames pipeline, sampling frames from video
        clip_len=1,  # Frames of each sampled output clip
        frame_interval=1,  # Temporal interval of adjacent sampled frames
        num_clips=3,  # Number of clips to be sampled
        test_mode=True),  # Whether to set test mode in sampling
    dict(  # Config of RawFrameDecode
        type='RawFrameDecode'),  # Load and decode Frames pipeline, picking raw frames with given indices
    dict(  # Config of Resize
        type='Resize',  # Resize pipeline
        scale=(-1, 256)),  # The scale to resize images
    dict(  # Config of CenterCrop
        type='CenterCrop',  # Center crop pipeline, cropping the center area from images
        crop_size=224),  # The size to crop images
    dict(  # Config of Flip
        type='Flip',  # Flip pipeline
        flip_ratio=0),  # Probability of implementing flip
    dict(  # Config of FormatShape
        type='FormatShape',  # Format shape pipeline, Format final image shape to the given input_format
        input_format='NCHW'),  # Final image shape format
    dict(type='PackActionInputs')  # Config of PackActionInputs
]
test_pipeline = [  # Testing data processing pipeline
    dict(  # Config of SampleFrames
        type='SampleFrames',  # Sample frames pipeline, sampling frames from video
        clip_len=1,  # Frames of each sampled output clip
        frame_interval=1,  # Temporal interval of adjacent sampled frames
        num_clips=25,  # Number of clips to be sampled
        test_mode=True),  # Whether to set test mode in sampling
    dict(  # Config of RawFrameDecode
        type='RawFrameDecode'),  # Load and decode Frames pipeline, picking raw frames with given indices
    dict(  # Config of Resize
        type='Resize',  # Resize pipeline
        scale=(-1, 256)),  # The scale to resize images
    dict(  # Config of TenCrop
        type='TenCrop',  # Ten crop pipeline, cropping ten area from images
        crop_size=224),  # The size to crop images
    dict(  # Config of Flip
        type='Flip',  # Flip pipeline
        flip_ratio=0),  # Probability of implementing flip
    dict(  # Config of FormatShape
        type='FormatShape',  # Format shape pipeline, Format final image shape to the given input_format
        input_format='NCHW'),  # Final image shape format
    dict(type='PackActionInputs')  # Config of PackActionInputs
]

train_dataloader = dict(  # Config of train dataloader
    batch_size=32,  # Batch size of each single GPU during training
    num_workers=8,  # Workers to pre-fetch data for each single GPU during training
    persistent_workers=True,  # If `True`, the dataloader will not shut down the worker processes after an epoch end, which can accelerate training speed
    sampler=dict(
        type='DefaultSampler',  # DefaultSampler which supports both distributed and non-distributed training. Refer to https://github.com/open-mmlab/mmengine/blob/main/mmengine/dataset/sampler.py
        shuffle=True),  # Randomly shuffle the training data in each epoch
    dataset=dict(  # Config of train dataset
        type=dataset_type,
        ann_file=ann_file_train,  # Path of annotation file
        data_prefix=dict(img=data_root),  # Prefix of frame path
        pipeline=train_pipeline))
val_dataloader = dict(  # Config of validation dataloader
    batch_size=1,  # Batch size of each single GPU during validation
    num_workers=8,  # Workers to pre-fetch data for each single GPU during validation
    persistent_workers=True,  # If `True`, the dataloader will not shut down the worker processes after an epoch end
    sampler=dict(
        type='DefaultSampler',
        shuffle=False),  # Not shuffle during validation and testing
    dataset=dict(  # Config of validation dataset
        type=dataset_type,
        ann_file=ann_file_val,  # Path of annotation file
        data_prefix=dict(img=data_root_val),  # Prefix of frame path
        pipeline=val_pipeline,
        test_mode=True))
test_dataloader = dict(  # Config of test dataloader
    batch_size=32,  # Batch size of each single GPU during testing
    num_workers=8,  # Workers to pre-fetch data for each single GPU during testing
    persistent_workers=True,  # If `True`, the dataloader will not shut down the worker processes after an epoch end
    sampler=dict(
        type='DefaultSampler',
        shuffle=False),  # Not shuffle during validation and testing
    dataset=dict(  # Config of test dataset
        type=dataset_type,
        ann_file=ann_file_val,  # Path of annotation file
        data_prefix=dict(img=data_root_val),  # Prefix of frame path
        pipeline=test_pipeline,
        test_mode=True))

# evaluation settings
val_evaluator = dict(type='AccMetric')  # Config of validation evaluator
test_evaluator = val_evaluator  # Config of testing evaluator

train_cfg = dict(  # Config of training loop
    type='EpochBasedTrainLoop',  # Name of training loop
    max_epochs=100,  # Total training epochs
    val_begin=1,  # The epoch that begins validating
    val_interval=1)  # Validation interval
val_cfg = dict(  # Config of validation loop
    type='ValLoop')  # Name of validation loop
test_cfg = dict( # Config of testing loop
    type='TestLoop')  # Name of testing loop

# learning policy
param_scheduler = [  # Parameter scheduler for updating optimizer parameters, support dict or list
    dict(type='MultiStepLR',  # Decays the learning rate once the number of epoch reaches one of the milestones
        begin=0,  # Step at which to start updating the learning rate
        end=100,  # Step at which to stop updating the learning rate
        by_epoch=True,  # Whether the scheduled learning rate is updated by epochs
        milestones=[40, 80],  # Steps to decay the learning rate
        gamma=0.1)]  # Multiplicative factor of learning rate decay

# optimizer
optim_wrapper = dict(  # Config of optimizer wrapper
    type='OptimWrapper',  # Name of optimizer wrapper, switch to AmpOptimWrapper to enable mixed precision training
    optimizer=dict(  # Config of optimizer. Support all kinds of optimizers in PyTorch. Refer to https://pytorch.org/docs/stable/optim.html#algorithms
        type='SGD',  # Name of optimizer
        lr=0.01,  # Learning rate
        momentum=0.9,  # Momentum factor
        weight_decay=0.0001),  # Weight decay
    clip_grad=dict(max_norm=40, norm_type=2))  # Config of gradient clip

# runtime settings
default_scope = 'mmaction'  # The default registry scope to find modules. Refer to https://mmengine.readthedocs.io/en/latest/tutorials/registry.html
default_hooks = dict(  # Hooks to execute default actions like updating model parameters and saving checkpoints.
    runtime_info=dict(type='RuntimeInfoHook'),  # The hook to updates runtime information into message hub
    timer=dict(type='IterTimerHook'),  # The logger used to record time spent during iteration
    logger=dict(
        type='LoggerHook',  # The logger used to record logs during training/validation/testing phase
        interval=20,  # Interval to print the log
        ignore_last=False), # Ignore the log of last iterations in each epoch
    param_scheduler=dict(type='ParamSchedulerHook'),  # The hook to update some hyper-parameters in optimizer
    checkpoint=dict(
        type='CheckpointHook',  # The hook to save checkpoints periodically
        interval=3,  # The saving period
        save_best='auto',  # Specified metric to mearsure the best checkpoint during evaluation
        max_keep_ckpts=3),  # The maximum checkpoints to keep
    sampler_seed=dict(type='DistSamplerSeedHook'),  # Data-loading sampler for distributed training
    sync_buffers=dict(type='SyncBuffersHook'))  # Synchronize model buffers at the end of each epoch
env_cfg = dict(  # Dict for setting environment
    cudnn_benchmark=False,  # Whether to enable cudnn benchmark
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0), # Parameters to setup multiprocessing
    dist_cfg=dict(backend='nccl')) # Parameters to setup distributed environment, the port can also be set

log_processor = dict(
    type='LogProcessor',  # Log processor used to format log information
    window_size=20,  # Default smooth interval
    by_epoch=True)  # Whether to format logs with epoch type
vis_backends = [  # List of visualization backends
    dict(type='LocalVisBackend')]  # Local visualization backend
visualizer = dict(  # Config of visualizer
    type='ActionVisualizer',  # Name of visualizer
    vis_backends=vis_backends)
log_level = 'INFO'  # The level of logging
load_from = None  # Load model checkpoint as a pre-trained model from a given path. This will not resume training.
resume = False  # Whether to resume from the checkpoint defined in `load_from`. If `load_from` is None, it will resume the latest checkpoint in the `work_dir`.
