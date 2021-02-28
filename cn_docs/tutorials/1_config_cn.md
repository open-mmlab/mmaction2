# 教程 1：如何编写配置文件

MMAction2 使用 python 文件作为配置文件，将模块设计与继承设计整合到 MMAction2 的配置文件系统中，方便用户进行各种实验。
MMAction2 提供的所有配置文件都放置在 `$MMAction2/configs` 文件夹下，用户可以通过运行命令
`python tools/analysis/print_config.py /PATH/TO/CONFIG` 来查看完整的配置信息，从而方便检查所对应的配置文件。

<!-- TOC -->

- [通过脚本参数修改配置信息]()
- [配置文件结构]()
- [配置文件命名规则]()
  - [时序动作检测的配置文件系统]()
  - [动作识别的配置文件系统]()
  - [时空动作检测的配置文件系统]()
- [常见问题]()
  - [配置文件中的中间变量]()

<!-- TOC -->

## 通过脚本参数修改配置信息

当用户使用脚本 "tools/train.py" 或者 "tools/test.py" 提交任务时，可以通过指定 `--cfg-options` 参数来直接修改所使用的配置文件内容。

- 更新配置文件内字典链中的键

  用户可以按照原始配置中的字典键顺序来指定配置文件的设置。
  例如，`--cfg-options model.backbone.norm_eval=False` 会改变 `train` 模式下模型骨架 backbone 中所有的 BN 模块。

- 更新配置文件内列表的键

  配置文件中的存在一些由字典组合成的列表。例如，训练流水线 `data.train.pipeline` 通常是一个 python 列表。
  类似，`[dict(type='SampleFrames'), ...]`。如果用户向更改其中的 `'SampleFrames'` 为 `'DenseSampleFrames'`，
  可以指定 `--cfg-options data.train.pipeline.0.type=DenseSampleFrames`。

- 更新列表/元组的值。

  当配置文件中需要更新的是一个列表或者元组，例如，配置文件通常会设置 `workflow=[('train', 1)]`，用户如果想更改，
  需要指定 `--cfg-options workflow="[(train,1),(val,1)]"`。注意这里的引号 \" 对于列表/元组数据类型的修改是必要的，
  并且 **不允许** 引号内所指定的值的书写存在空格。

## 配置文件结构

在 `config/_base_` 文件夹下存在 3 种基本组件类型： 模型（model）, 调度（schedule）, 默认运行设置（default_runtime）。
许多方法都可以方便地通过组合这些组件进行实现，如 TSN，I3D，SlowOnly，等。
其中，通过 `_base_` 下组件来构建的配置被称为 _原始配置_（_primitive_）。

对于在同一文件夹下的所有配置文件，MMAction2 推荐只在其中存在 **一个** _原始配置_ 文件。
所有其他的配置文件都应该继承 _原始配置_ 文件，这样就能保证配置文件的最大继承等级为3。

为了方便理解，MMAction2 推荐用户继承现有的方法的配置文件。
例如，如果配置的修改基于 TSN，用户首先应通过指定 `_base_ = ../tsn/tsn_r50_1x1x3_100e_kinetics400_rgb.py` 继承 TSN 的基本结构，
并修改其中必要的内容以完成继承。

如果用户实现一个独立于任何一个现有的方法结构的新方法，则需要在 `configs/TASK` 建立新的文件夹。

更多详细内容，请参考 [mmcv](https://mmcv.readthedocs.io/en/latest/utils.html#config)。

## 配置文件命名规则

MMAction2 按照以下风格进行配置文件命名，代码库的贡献者需要遵循相同的命名规则。

```
{model}_[model setting]_{backbone}_[misc]_{data setting}_[gpu x batch_per_gpu]_{schedule}_{dataset}_{modality}
```

其中，`{xxx}` 表示必要的命名域，`[yyy]` 表示可选的命名域。

- `{model}`：模型类型，如 `tsn`，`i3d` 等。
- `[model setting]`：一些模型上的特殊设置。
- `{backbone}`：骨架类型，如 `r50`（ResNet-50）等。
- `[misc]`：模型的额外设置或插件，如 `dense`，`320p`，`video`等。
- `{data setting}`：采桢数据格式，形如 `{clip_len}x{frame_interval}x{num_clips}`。
- `[gpu x batch_per_gpu]`：GPU 数量以及每个 GPU 上的采样。
- `{schedule}`：训练时的调度设置，如 `20e` 表示 20 个周期（epoch）。
- `{dataset}`：数据集名，如 `kinetics400`，`mmit`等。
- `{modality}`：帧的模态，如 `rgb`, `flow`等。

### 时序动作检测的配置文件系统

MMAction2 将模块化设计整合到配置文件系统中，以便于执行各种不同的实验。

- 以 BMN 为例

    为了帮助用户理解 MMAction2 的完整配置文件结构，及时序动作检测系统中的一些模块，这里以 BMN 为例对其配置文件进行了注释。
    对于每个模块的详细用法以及对应参数的选择，请参照 API 文档。

    ```python
    # 模型设置
    model = dict(  # 模型的配置
        type='BMN',  # 时序动作检测器的类型
        temporal_dim=100,  # 每个视频中所选择的帧数量
        boundary_ratio=0.5,  # 视频边界的决策几率
        num_samples=32,  # 每个候选的采样数
        num_samples_per_bin=3,  # 每个样本的直方图采样数
        feat_dim=400,  # 特征维度
        soft_nms_alpha=0.4,  # 软 NMS 的 alpha 值
        soft_nms_low_threshold=0.5,  # 软 NMS 的下界
        soft_nms_high_threshold=0.9,  # 软 NMS 的上届
        post_process_top_k=100)  # 后处理得到的最好的 K 个proposal
    # 模型训练和测试的设置
    train_cfg = None  # 训练 BMN 的超参配置
    test_cfg = dict(average_clips='score')  # 测试 BMN 的超参配置

    # 数据集设置
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
        # which are builed on `constructor`, referring to "tutorials/5_new_modules.md"
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
