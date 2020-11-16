# Config System

We use python files as configs. You can find all the provided configs under `$MMAction2/configs`.

<!-- TOC -->

- [Config File Naming Convention](#config-file-naming-convention)
- [Config File Structure](#config-file-structure)
  * [Config System for Action localization](#config-system-for-action-localization)
  * [Config System for Action Recognition](#config-system-for-action-recognition)
- [FAQ](#faq)
  * [Use intermediate variables in configs](#use-intermediate-variables-in-configs)

<!-- TOC -->

## Config File Naming Convention

We follow the style below to name config files. Contributors are advised to follow the same style.

```
{model}_[model setting]_{backbone}_[misc]_{data setting}_[gpu x batch_per_gpu]_{schedule}_{dataset}_{modality}
```

`{xxx}` is required field and `[yyy]` is optional.

- `{model}`: model type, e.g. `tsn`, `i3d`, etc.
- `[model setting]`: specific setting for some models.
- `{backbone}`: backbone type, e.g. `r50` (ResNet-50), etc.
- `[misc]`: miscellaneous setting/plugins of model, e.g. `dense`, `320p`, `video`, etc.
- `{data setting}`: frame sample setting in `{clip_len}x{frame_interval}x{num_clips}` format.
- `[gpu x batch_per_gpu]`: GPUs and samples per GPU.
- `{schedule}`: training schedule, e.g. `20e` means 20 epochs.
- `{dataset}`: dataset name, e.g. `kinetics400`, `mmit`, etc.
- `{modality}`: frame modality, e.g. `rgb`, `flow`, etc.

## Config File Structure

Please refer to the corresponding pages for config file structure for different tasks.

### Config System for Action localization

We incorporate modular design into our config system,
which is convenient to conduct various experiments.

- An Example of BMN

    To help the users have a basic idea of a complete config structure and the modules in an action localization system,
    we make brief comments on the config of BMN as the following.
    For more detailed usage and alternative for per parameter in each module, please refer to the API documentation.

    ```python
    # model settings
    model = dict(  # Config of the model
        type='BMN',  # Type of the localizer
        temporal_dim=100,  # Total frames selected for each video
        boundary_ratio=0.5,  # Ratio for determining video boundaries
        num_samples=32,  # Number of samples for each proposal
        num_samples_per_bin=3,  # Number of bin samples for each sample
        feat_dim=400,  # Dimension of feature
        soft_nms_alpha=0.4,  # Soft NMS alpha
        soft_nms_low_threshold=0.5,  # Soft NMS low threshold
        soft_nms_high_threshold=0.9,  # Soft NMS high threshold
        post_process_top_k=100)  # Top k proposals in post process
    # model training and testing settings
    train_cfg = None  # Config of training hyperparameters for BMN
    test_cfg = dict(average_clips='score')  # Config for testing hyperparameters for BMN

    # dataset settings
    dataset_type = 'ActivityNetDataset'  # Type of dataset for training, valiation and testing
    data_root = 'data/activitynet_feature_cuhk/csv_mean_100/'  # Root path to data for training
    data_root_val = 'data/activitynet_feature_cuhk/csv_mean_100/'  # Root path to data for validation and testing
    ann_file_train = 'data/ActivityNet/anet_anno_train.json'  # Path to the annotation file for training
    ann_file_val = 'data/ActivityNet/anet_anno_val.json'  # Path to the annotation file for validation
    ann_file_test = 'data/ActivityNet/anet_anno_test.json'  # Path to the annotation file for testing

    train_pipeline = [  # List of training pipeline steps
        dict(type='LoadLocalizationFeature'),  # Load localization feature pipeline
        dict(type='GenerateLocalizationLabels'),  # Generate localization labels pipeline
        dict(  # Config of Collect
            type='Collect',  # Collect pipeline that decides which keys in the data should be passed to the localizer
            keys=['raw_feature', 'gt_bbox'],  # Keys of input
            meta_name='video_meta',  # Meta name
            meta_keys=['video_name']),  # Meta keys of input
        dict(  # Config of ToTensor
            type='ToTensor',  # Convert other types to tensor type pipeline
            keys=['raw_feature']),  # Keys to be converted from image to tensor
        dict(  # Config of ToDataContainer
            type='ToDataContainer',  # Pipeline to convert the data to DataContainer
            fields=[dict(key='gt_bbox', stack=False, cpu_only=True)])  # Required fields to be converted with keys and attributes
    ]
    val_pipeline = [  # List of validation pipeline steps
        dict(type='LoadLocalizationFeature'),  # Load localization feature pipeline
        dict(type='GenerateLocalizationLabels'),  # Generate localization labels pipeline
        dict(  # Config of Collect
            type='Collect',  # Collect pipeline that decides which keys in the data should be passed to the localizer
            keys=['raw_feature', 'gt_bbox'],  # Keys of input
            meta_name='video_meta',  # Meta name
            meta_keys=[
                'video_name', 'duration_second', 'duration_frame', 'annotations',
                'feature_frame'
            ]),  # Meta keys of input
        dict(  # Config of ToTensor
            type='ToTensor',  # Convert other types to tensor type pipeline
            keys=['raw_feature']),  # Keys to be converted from image to tensor
        dict(  # Config of ToDataContainer
            type='ToDataContainer',  # Pipeline to convert the data to DataContainer
            fields=[dict(key='gt_bbox', stack=False, cpu_only=True)])  # Required fields to be converted with keys and attributes
    ]
    test_pipeline = [  # List of testing pipeline steps
        dict(type='LoadLocalizationFeature'),  # Load localization feature pipeline
        dict(  # Config of Collect
            type='Collect',  # Collect pipeline that decides which keys in the data should be passed to the localizer
            keys=['raw_feature'],  # Keys of input
            meta_name='video_meta',  # Meta name
            meta_keys=[
                'video_name', 'duration_second', 'duration_frame', 'annotations',
                'feature_frame'
            ]),  # Meta keys of input
        dict(  # Config of ToTensor
            type='ToTensor',  # Convert other types to tensor type pipeline
            keys=['raw_feature']),  # Keys to be converted from image to tensor
    ]
    data = dict(  # Config of data
        videos_per_gpu=8,  # Batch size of each single GPU
        workers_per_gpu=8,  # Workers to pre-fetch data for each single GPU
        train_dataloader=dict(  # Additional config of train dataloader
            drop_last=True),  # Whether to drop out the last batch of data in training
        val_dataloader=dict(  # Additional config of validation dataloader
            videos_per_gpu=1),  # Batch size of each single GPU during evaluation
        test_dataloader=dict(  # Additional config of test dataloader
            videos_per_gpu=2),  # Batch size of each single GPU during testing
        test=dict(  # Testing dataset config
            type=dataset_type,
            ann_file=ann_file_test,
            pipeline=test_pipeline,
            data_prefix=data_root_val),
        val=dict(  # Validation dataset config
            type=dataset_type,
            ann_file=ann_file_val,
            pipeline=val_pipeline,
            data_prefix=data_root_val),
        train=dict(  # Training dataset config
            type=dataset_type,
            ann_file=ann_file_train,
            pipeline=train_pipeline,
            data_prefix=data_root))

    # optimizer
    optimizer = dict(
        # Config used to build optimizer, support (1). All the optimizers in PyTorch
        # whose arguments are also the same as those in PyTorch. (2). Custom optimizers
        # which are builed on `constructor`, referring to "tutorials/new_modules.md"
        # for implementation.
        type='Adam',  # Type of optimizer, refer to https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/optimizer/default_constructor.py#L13 for more details
        lr=0.001,  # Learning rate, see detail usages of the parameters in the documentaion of PyTorch
        weight_decay=0.0001)  # Weight decay of Adam
    optimizer_config = dict(  # Config used to build the optimizer hook
        grad_clip=None)  # Most of the methods do not use gradient clip
    # learning policy
    lr_config = dict(  # Learning rate scheduler config used to register LrUpdater hook
        policy='step',  # Policy of scheduler, also support CosineAnnealing, Cyclic, etc. Refer to details of supported LrUpdater from https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py#L9
        step=7)  # Steps to decay the learning rate

    total_epochs = 9  # Total epochs to train the model
    checkpoint_config = dict(  # Config to set the checkpoint hook, Refer to https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint.py for implementation
        interval=1)  # Interval to save checkpoint
    evaluation = dict(  # Config of evaluation during training
        interval=1,  # Interval to perform evaluation
        metrics=['AR@AN'])  # Metrics to be performed
    log_config = dict(  # Config to register logger hook
        interval=50,  # Interval to print the log
        hooks=[  # Hooks to be implemented during training
            dict(type='TextLoggerHook'),  # The logger used to record the training process
            # dict(type='TensorboardLoggerHook'),  # The Tensorboard logger is also supported
        ])

    # runtime settings
    dist_params = dict(backend='nccl')  # Parameters to setup distributed training, the port can also be set
    log_level = 'INFO'  # The level of logging
    work_dir = './work_dirs/bmn_400x100_2x8_9e_activitynet_feature/'  # Directory to save the model checkpoints and logs for the current experiments
    load_from = None  # load models as a pre-trained model from a given path. This will not resume training
    resume_from = None  # Resume checkpoints from a given path, the training will be resumed from the epoch when the checkpoint's is saved
    workflow = [('train', 1)]  # Workflow for # runner. [('train', 1)] means there is only one workflow and the workflow named 'train' is executed once
    output_config = dict(  # Config of localization ouput
        out=f'{work_dir}/results.json',  # Path to output file
        output_format='json')  # File format of output file
    ```

### Config System for Action Recognition

We incorporate modular design into our config system,
which is convenient to conduct various experiments.

- An Example of TSN

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
    ann_file_train = 'data/kinetics400/kinetics400_train_list_rawframes.txt'  # Path to the annotation file for training
    ann_file_val = 'data/kinetics400/kinetics400_val_list_rawframes.txt'  # Path to the annotation file for validation
    ann_file_test = 'data/kinetics400/kinetics400_val_list_rawframes.txt'  # Path to the annotation file for testing
    img_norm_cfg = dict(  # Config of image normalition used in data pipeline
        mean=[123.675, 116.28, 103.53],  # Mean values of different channels to normalize
        std=[58.395, 57.12, 57.375],  # Std values of different channels to normalize
        to_bgr=False)  # Whether to convert channels from RGB to BGR

    train_pipeline = [  # List of training pipeline steps
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
        dict(  # Config of RawFrameDecode
            type='RawFrameDecode'),  # Load and decode Frames pipeline, picking raw frames with given indices
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
        train_dataloader=dict(  # Additional config of train dataloader
            drop_last=True),  # Whether to drop out the last batch of data in training
        val_dataloader=dict(  # Additional config of validation dataloader
            videos_per_gpu=1),  # Batch size of each single GPU during evaluation
        test_dataloader=dict(  # Additional config of test dataloader
            videos_per_gpu=2),  # Batch size of each single GPU during testing
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
        policy='step',  # Policy of scheduler, also support CosineAnnealing, Cyclic, etc. Refer to details of supported LrUpdater from https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py#L9
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


## FAQ

### Use intermediate variables in configs

Some intermediate variables are used in the config files, like `train_pipeline`/`val_pipeline`/`test_pipeline`,
`ann_file_train`/`ann_file_val`/`ann_file_test`, `img_norm_cfg` etc.

For Example, we would like to first define `train_pipeline`/`val_pipeline`/`test_pipeline` and pass them into `data`.
Thus, `train_pipeline`/`val_pipeline`/`test_pipeline` are intermediate variable.

we also define `ann_file_train`/`ann_file_val`/`ann_file_test` and `data_root`/`data_root_val` to provide data pipeline some
basic information.

In addition, we use `img_norm_cfg` as intermediate variables to construct data augmentation components.

```python
...
dataset_type = 'RawframeDataset'
data_root = 'data/kinetics400/rawframes_train'
data_root_val = 'data/kinetics400/rawframes_val'
ann_file_train = 'data/kinetics400/kinetics400_train_list_rawframes.txt'
ann_file_val = 'data/kinetics400/kinetics400_val_list_rawframes.txt'
ann_file_test = 'data/kinetics400/kinetics400_val_list_rawframes.txt'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

train_pipeline = [
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.8),
        random_crop=False,
        max_wh_scale_gap=0),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=2,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=2,
        num_clips=10,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]

data = dict(
    videos_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=test_pipeline))
```
