_base_ = [
    '../../_base_/models/tsn_r50.py', '../../_base_/schedules/sgd_100e.py',
    '../../_base_/default_runtime.py'
]

# model settings
model = dict(
    cls_head=dict(
        type='TSNHead',
        num_classes=6  # change from 400 to 101
        ))

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

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=50,  # change from 100 to 50
    val_begin=1,
    val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning policy
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=50,  # change from 100 to 50
        by_epoch=True,
        milestones=[20, 40],  # change milestones
        gamma=0.1)
]

# optimizer
optim_wrapper = dict(
    optimizer=dict(
        type='SGD',
        lr=0.005, # change from 0.01 to 0.005
        momentum=0.9,
        weight_decay=0.0001),
    clip_grad=dict(max_norm=40, norm_type=2))

# val_evaluator = dict(type='AccMetric')
# test_evaluator = val_evaluator

default_hooks = dict(checkpoint=dict(interval=3, max_keep_ckpts=3))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (32 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=256)

# use the pre-trained model for the whole TSN network
load_from = 'https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb_20220906-cd10898e.pth'
visualizer=dict(type='Visualizer', vis_backends=[dict(type='WandbVisBackend')])
