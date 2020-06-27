# Config System for Action Recognition

We incorporate modular and inheritance design into our config system,
which is convenient to conduct various experiments.

## An Example of TSN

To help the users have a basic idea of a complete config structure and the modules in an action recognition system,
we make brief comments on the config of TSN as the following.
For more detailed usage and alternative for per parameter in each module, please refer to the API documentation.

```python
# model settings
model = dict(  # Config of the model
    type='Recognizer2D',  # Type of the recognizer
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
        init_std=0.01))  # Std value for linear layer initiation
# model training and testing settings
train_cfg = None  # Config of training hyperparameters for TSN
test_cfg = dict(average_clips=None) # Config for testing hyperparameters for TSN. Here we define clip averaging method in it

# dataset settings
dataset_type = 'RawframeDataset'  # Type of dataset for training, valiation and testing
data_root = 'data/kinetics400/rawframes_train/'  # Root path to data for training
data_root_val = 'data/kinetics400/rawframes_val/'  # Root path to data for validation and testing
ann_file_train = 'data/kinetics400/kinetics_train_list.txt'  # Path to the annotation file for training
ann_file_val = 'data/kinetics400/kinetics_val_list.txt'  # Path to the annotation file for validation
ann_file_test = 'data/kinetics400/kinetics_val_list.txt'  # Path to the annotation file for testing
img_norm_cfg = dict(  # Config of image normalition used in data pipeline
    mean=[123.675, 116.28, 103.53],  # Mean values of different channels to normalize
    std=[58.395, 57.12, 57.375],  # Std values of different channels to normalize
    to_bgr=False)  # Whether to convert channels from RGB to BGR
mc_cfg = dict(  # Config of memcached setting
    server_list_cfg='/mnt/lustre/share/memcached_client/server_list.conf',  # Path to server list config
    client_cfg='/mnt/lustre/share/memcached_client/client.conf',  # Path to client config
    sys_path='/mnt/lustre/share/pymc/py3')  # Path to `pymc` in python3 version
train_pipeline = [  # List of training pipeline steps
    dict(  # Config of SampleFrames
        type='SampleFrames',  # Sample frames pipeline, sampling frames from video
        clip_len=1,  # Frames of each sampled output clip
        frame_interval=1,  # Temporal interval of adjacent sampled frames
        num_clips=3),  # Number of clips to be sampled
    dict(  # Config of FrameSelector
        type='FrameSelector',  # Frame selector pipeline, selecting raw frames with given indices
        io_backend='memcached',  # Storage backend type
        **mc_cfg),  # Config of memcached
    dict(  # Config of Resize
        type='Resize',  # Resize pipeline
        scale=(-1, 256)),  # The scale to resize images
    dict(  # Config of MultiScaleCrop
        type='MultiScaleCrop',  # Multi scale crop pipeline, cropping images with a list of randomly selected scales
        input_size=224,  # Input size of the network
        scales=(1, 0.875, 0.75, 0.66),  # Scales of weight and height to be selected
        random_crop=False,  # Whether to randomly sample cropping bbox
        max_wh_scale_gap=1),  # Maximum gap of w and h scale levels
    dict(  # Config of Resize
        type='Resize',  # Resize pipeline
        scale=(224, 224),  # The scale to resize images
        keep_ratio=False),  # Whether to resize with changing the aspect ratio
    dict(  # Config of Flip
        type='Flip',  # Flip Pipeline
        flip_ratio=0.5),  # Probability of implementing flip
    dict(  # Config of Normalize
        type='Normalize',  # Normalize pipeline
        **img_norm_cfg),  # Config of image normalization
    dict(  # Config of FormatShape
        type='FormatShape',  # Format shape pipeline, Format final image shape to the given input_format
        input_format='NCHW'),  # Final image shape format
    dict(  # Config of Collect
        type='Collect',  # Collect pipeline that decides which keys in the data should be passed to the recognizer
        keys=['imgs', 'label'],  # Keys of input
        meta_keys=[]),  # Meta keys of input
    dict(  # Config of ToTensor
        type='ToTensor',  # Convert other types to tensor type pipeline
        keys=['imgs', 'label'])  # Keys to be converted from image to tensor
]
val_pipeline = [  # List of validation pipeline steps
    dict(  # Config of SampleFrames
        type='SampleFrames',  # Sample frames pipeline, sampling frames from video
        clip_len=1,  # Frames of each sampled output clip
        frame_interval=1,  # Temporal interval of adjacent sampled frames
        num_clips=3,  # Number of clips to be sampled
        test_mode=True),  # Whether to set test mode in sampling
    dict(  # Config of FrameSelector
        type='FrameSelector',  # Frame selector pipeline, selecting raw frames with given indices
        io_backend='memcached',  # Storage backend type
        **mc_cfg),  # Config of memcached
    dict(  # Config of Resize
        type='Resize',  # Resize pipeline
        scale=(-1, 256)),  # The scale to resize images
    dict(  # Config of CenterCrop
        type='CenterCrop',  # Center crop pipeline, cropping the center area from images
        crop_size=224),  # The size to crop images
    dict(  # Config of Flip
        type='Flip',  # Flip pipeline
        flip_ratio=0),  # Probability of implementing flip
    dict(  # Config of Normalize
        type='Normalize',  # Normalize pipeline
        **img_norm_cfg),  # Config of image normalization
    dict(  # Config of FormatShape
        type='FormatShape',  # Format shape pipeline, Format final image shape to the given input_format
        input_format='NCHW'),  # Final image shape format
    dict(  # Config of Collect
        type='Collect',  # Collect pipeline that decides which keys in the data should be passed to the recognizer
        keys=['imgs', 'label'],  # Keys of input
        meta_keys=[]),  # Meta keys of input
    dict(  # Config of ToTensor
        type='ToTensor',  # Convert other types to tensor type pipeline
        keys=['imgs'])  # Keys to be converted from image to tensor
]
test_pipeline = [  # List of testing pipeline steps
    dict(  # Config of SampleFrames
        type='SampleFrames',  # Sample frames pipeline, sampling frames from video
        clip_len=1,  # Frames of each sampled output clip
        frame_interval=1,  # Temporal interval of adjacent sampled frames
        num_clips=25,  # Number of clips to be sampled
        test_mode=True),  # Whether to set test mode in sampling
    dict(  # Config of FrameSelector
        type='FrameSelector',  # Frame selector pipeline, selecting raw frames with given indices
        io_backend='memcached',  # Storage backend type
        **mc_cfg),  # Config of memcached
    dict(  # Config of Resize
        type='Resize',  # Resize pipeline
        scale=(-1, 256)),  # The scale to resize images
    dict(  # Config of CenterCrop
        type='TenCrop',  # Center crop pipeline, cropping the center area from images
        crop_size=224),  # The size to crop images
    dict(  # Config of Flip
        type='Flip',  # Flip pipeline
        flip_ratio=0),  # Probability of implementing flip
    dict(  # Config of Normalize
        type='Normalize',  # Normalize pipeline
        **img_norm_cfg),  # Config of image normalization
    dict(  # Config of FormatShape
        type='FormatShape',  # Format shape pipeline, Format final image shape to the given input_format
        input_format='NCHW'),  # Final image shape format
    dict(  # Config of Collect
        type='Collect',  # Collect pipeline that decides which keys in the data should be passed to the recognizer
        keys=['imgs', 'label'],  # Keys of input
        meta_keys=[]),  # Meta keys of input
    dict(  # Config of ToTensor
        type='ToTensor',  # Convert other types to tensor type pipeline
        keys=['imgs'])  # Keys to be converted from image to tensor
]
data = dict(  # Config of data
    videos_per_gpu=32,  # Batch size of each single GPU
    workers_per_gpu=4,  # Workers to pre-fetch data for each single GPU
    train=dict(  # Training dataset config
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(  # Validation dataset config
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline),
    test=dict(  # Testing dataset config
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline))
# optimizer
optimizer = dict(
    # Config used to build optimizer, support (1). All the optimizers in PyTorch
    # whose arguments are also the same as those in PyTorch. (2). Custom optimizers
    # which are builed on `constructor`, referring to "tutorials/new_modules.md"
    # for implementation.
    type='SGD',  # Type of optimizer, refer to https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/optimizer/default_constructor.py#L13 for more details
    lr=0.01,  # Learning rate, see detail usages of the parameters in the documentaion of PyTorch
    momentum=0.9,  # Momentum,
    weight_decay=0.0001)  # Weight decay of SGD
optimizer_config = dict(  # Config used to build the optimizer hook
    grad_clip=dict(max_norm=40, norm_type=2))  # Use gradient clip
# learning policy
lr_config = dict(  # Learning rate scheduler config used to register LrUpdater hook
    policy='step',  # Policy of scheduler, also support CosineAnealing, Cyclic, etc. Refer to details of supported LrUpdater from https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py#L9
    step=[40, 80])  # Steps to decay the learning rate
total_epochs = 100  # Total epochs to train the model
checkpoint_config = dict(  # Config to set the checkpoint hook, Refer to https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint.py for implementation
    interval=5)  # Interval to save checkpoint
evaluation = dict(  # Config of evaluation during training
    interval=5,  # Interval to perform evaluation
    metrics=['top_k_accuracy', 'mean_class_accuracy'],  # Metrics to be performed
    topk=(1, 5))  # K value for `top_k_accuracy` metric
log_config = dict(  # Config to register logger hook
    interval=20,  # Interval to print the log
    hooks=[  # Hooks to be implemented during training
        dict(type='TextLoggerHook'),  # The logger used to record the training process
        # dict(type='TensorboardLoggerHook'),  # The Tensorboard logger is also supported
    ])

# runtime settings
dist_params = dict(backend='nccl')  # Parameters to setup distributed training, the port can also be set
log_level = 'INFO'  # The level of logging
work_dir = './work_dirs/tsn_r50_1x1x3_100e_kinetics400_rgb/'  # Directory to save the model checkpoints and logs for the current experiments
load_from = None  # load models as a pre-trained model from a given path. This will not resume training
resume_from = None  # Resume checkpoints from a given path, the training will be resumed from the epoch when the checkpoint's is saved
workflow = [('train', 1)]  # Workflow for runner. [('train', 1)] means there is only one workflow and the workflow named 'train' is executed once

```
