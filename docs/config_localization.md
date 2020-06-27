# Config System for Action localization

We incorporate modular and inheritance design into our config system,
which is convenient to conduct various experiments.

## An Example of BMN

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
    train_dataloader=dict(  # Addition config of train dataloader
        drop_last=True),  # Whether to drop out the last batch of data in training
    val_dataloader=dict(  # Addition config of validation dataloader
        videos_per_gpu=1),  # Batch size of each single GPU during evaluation
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
    policy='step',  # Policy of scheduler, also support CosineAnealing, Cyclic, etc. Refer to details of supported LrUpdater from https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py#L9
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
