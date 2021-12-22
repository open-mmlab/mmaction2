# 教程 1：如何编写配置文件

MMAction2 使用 python 文件作为配置文件。其配置文件系统的设计将模块化与继承整合进来，方便用户进行各种实验。
MMAction2 提供的所有配置文件都放置在 `$MMAction2/configs` 文件夹下，用户可以通过运行命令
`python tools/analysis/print_config.py /PATH/TO/CONFIG` 来查看完整的配置信息，从而方便检查所对应的配置文件。

<!-- TOC -->

- [通过命令行参数修改配置信息](#通过命令行参数修改配置信息)
- [配置文件结构](#配置文件结构)
- [配置文件命名规则](#配置文件命名规则)
  - [时序动作检测的配置文件系统](#时序动作检测的配置文件系统)
  - [动作识别的配置文件系统](#动作识别的配置文件系统)
  - [时空动作检测的配置文件系统](#时空动作检测的配置文件系统)
- [常见问题](#常见问题)
  - [配置文件中的中间变量](#配置文件中的中间变量)

<!-- TOC -->

## 通过命令行参数修改配置信息

当用户使用脚本 "tools/train.py" 或者 "tools/test.py" 提交任务时，可以通过指定 `--cfg-options` 参数来直接修改所使用的配置文件内容。

- 更新配置文件内的字典

  用户可以按照原始配置中的字典键顺序来指定配置文件的设置。
  例如，`--cfg-options model.backbone.norm_eval=False` 会改变 `train` 模式下模型主干网络 backbone 中所有的 BN 模块。

- 更新配置文件内列表的键

  配置文件中，存在一些由字典组成的列表。例如，训练数据前处理流水线 data.train.pipeline 就是 python 列表。
  如，`[dict(type='SampleFrames'), ...]`。如果用户想更改其中的 `'SampleFrames'` 为 `'DenseSampleFrames'`，
  可以指定 `--cfg-options data.train.pipeline.0.type=DenseSampleFrames`。

- 更新列表/元组的值。

  当配置文件中需要更新的是一个列表或者元组，例如，配置文件通常会设置 `workflow=[('train', 1)]`，用户如果想更改，
  需要指定 `--cfg-options workflow="[(train,1),(val,1)]"`。注意这里的引号 \" 对于列表/元组数据类型的修改是必要的，
  并且 **不允许** 引号内所指定的值的书写存在空格。

## 配置文件结构

在 `config/_base_` 文件夹下存在 3 种基本组件类型： 模型（model）, 训练策略（schedule）, 运行时的默认设置（default_runtime）。
许多方法都可以方便地通过组合这些组件进行实现，如 TSN，I3D，SlowOnly 等。
其中，通过 `_base_` 下组件来构建的配置被称为 _原始配置_（_primitive_）。

对于在同一文件夹下的所有配置文件，MMAction2 推荐只存在 **一个** 对应的 _原始配置_ 文件。
所有其他的配置文件都应该继承 _原始配置_ 文件，这样就能保证配置文件的最大继承深度为 3。

为了方便理解，MMAction2 推荐用户继承现有方法的配置文件。
例如，如需修改 TSN 的配置文件，用户应先通过 `_base_ = '../tsn/tsn_r50_1x1x3_100e_kinetics400_rgb.py'` 继承 TSN 配置文件的基本结构，
并修改其中必要的内容以完成继承。

如果用户想实现一个独立于任何一个现有的方法结构的新方法，则需要像 `configs/recognition`, `configs/detection` 等一样，在 `configs/TASK` 中建立新的文件夹。

更多详细内容，请参考 [mmcv](https://mmcv.readthedocs.io/en/latest/utils.html#config)。

## 配置文件命名规则

MMAction2 按照以下风格进行配置文件命名，代码库的贡献者需要遵循相同的命名规则。

```
{model}_[model setting]_{backbone}_[misc]_{data setting}_[gpu x batch_per_gpu]_{schedule}_{dataset}_{modality}
```

其中，`{xxx}` 表示必要的命名域，`[yyy]` 表示可选的命名域。

- `{model}`：模型类型，如 `tsn`，`i3d` 等。
- `[model setting]`：一些模型上的特殊设置。
- `{backbone}`：主干网络类型，如 `r50`（ResNet-50）等。
- `[misc]`：模型的额外设置或插件，如 `dense`，`320p`，`video`等。
- `{data setting}`：采帧数据格式，形如 `{clip_len}x{frame_interval}x{num_clips}`。
- `[gpu x batch_per_gpu]`：GPU 数量以及每个 GPU 上的采样。
- `{schedule}`：训练策略设置，如 `20e` 表示 20 个周期（epoch）。
- `{dataset}`：数据集名，如 `kinetics400`，`mmit`等。
- `{modality}`：帧的模态，如 `rgb`, `flow`等。

### 时序动作检测的配置文件系统

MMAction2 将模块化设计整合到配置文件系统中，以便于执行各种不同的实验。

- 以 BMN 为例

    为了帮助用户理解 MMAction2 的配置文件结构，以及时序动作检测系统中的一些模块，这里以 BMN 为例，给出其配置文件的注释。
    对于每个模块的详细用法以及对应参数的选择，请参照 [API 文档](https://mmaction2.readthedocs.io/en/latest/api.html)。

    ```python
    # 模型设置
    model = dict(  # 模型的配置
        type='BMN',  # 时序动作检测器的类型
        temporal_dim=100,  # 每个视频中所选择的帧数量
        boundary_ratio=0.5,  # 视频边界的决策几率
        num_samples=32,  # 每个候选的采样数
        num_samples_per_bin=3,  # 每个样本的直方图采样数
        feat_dim=400,  # 特征维度
        soft_nms_alpha=0.4,  # soft-NMS 的 alpha 值
        soft_nms_low_threshold=0.5,  # soft-NMS 的下界
        soft_nms_high_threshold=0.9,  # soft-NMS 的上界
        post_process_top_k=100)  # 后处理得到的最好的 K 个 proposal
    # 模型训练和测试的设置
    train_cfg = None  # 训练 BMN 的超参配置
    test_cfg = dict(average_clips='score')  # 测试 BMN 的超参配置

    # 数据集设置
    dataset_type = 'ActivityNetDataset'  # 训练，验证，测试的数据集类型
    data_root = 'data/activitynet_feature_cuhk/csv_mean_100/'  # 训练集的根目录
    data_root_val = 'data/activitynet_feature_cuhk/csv_mean_100/'  # 验证集和测试集的根目录
    ann_file_train = 'data/ActivityNet/anet_anno_train.json'  # 训练集的标注文件
    ann_file_val = 'data/ActivityNet/anet_anno_val.json'  # 验证集的标注文件
    ann_file_test = 'data/ActivityNet/anet_anno_test.json'  # 测试集的标注文件

    train_pipeline = [  # 训练数据前处理流水线步骤组成的列表
        dict(type='LoadLocalizationFeature'),  # 加载时序动作检测特征
        dict(type='GenerateLocalizationLabels'),  # 生成时序动作检测标签
        dict(  # Collect 类的配置
            type='Collect',  # Collect 类决定哪些键会被传递到时序检测器中
            keys=['raw_feature', 'gt_bbox'],  # 输入的键
            meta_name='video_meta',  # 元名称
            meta_keys=['video_name']),  # 输入的元键
        dict(  # ToTensor 类的配置
            type='ToTensor',  # ToTensor 类将其他类型转化为 Tensor 类型
            keys=['raw_feature']),  # 将被从其他类型转化为 Tensor 类型的特征
        dict(  # ToDataContainer 类的配置
            type='ToDataContainer',  # 将一些信息转入到 ToDataContainer 中
            fields=[dict(key='gt_bbox', stack=False, cpu_only=True)])  # 携带额外键和属性的信息域
    ]
    val_pipeline = [  # 验证数据前处理流水线步骤组成的列表
        dict(type='LoadLocalizationFeature'),  # 加载时序动作检测特征
        dict(type='GenerateLocalizationLabels'),  # 生成时序动作检测标签
        dict(  # Collect 类的配置
            type='Collect',  # Collect 类决定哪些键会被传递到时序检测器中
            keys=['raw_feature', 'gt_bbox'],  # 输入的键
            meta_name='video_meta',  # 元名称
            meta_keys=[
                'video_name', 'duration_second', 'duration_frame', 'annotations',
                'feature_frame'
            ]),  # 输入的元键
        dict(  # ToTensor 类的配置
            type='ToTensor',  # ToTensor 类将其他类型转化为 Tensor 类型
            keys=['raw_feature']),  # 将被从其他类型转化为 Tensor 类型的特征
        dict(  # ToDataContainer 类的配置
            type='ToDataContainer',  # 将一些信息转入到 ToDataContainer 中
            fields=[dict(key='gt_bbox', stack=False, cpu_only=True)])  # 携带额外键和属性的信息域
    ]
    test_pipeline = [  # 测试数据前处理流水线步骤组成的列表
        dict(type='LoadLocalizationFeature'),  # 加载时序动作检测特征
        dict(  # Collect 类的配置
            type='Collect',  # Collect 类决定哪些键会被传递到时序检测器中
            keys=['raw_feature'],  # 输入的键
            meta_name='video_meta',  # 元名称
            meta_keys=[
                'video_name', 'duration_second', 'duration_frame', 'annotations',
                'feature_frame'
            ]),  # 输入的元键
        dict(  # ToTensor 类的配置
            type='ToTensor',  # ToTensor 类将其他类型转化为 Tensor 类型
            keys=['raw_feature']),  # 将被从其他类型转化为 Tensor 类型的特征
    ]
    data = dict(  # 数据的配置
        videos_per_gpu=8,  # 单个 GPU 的批大小
        workers_per_gpu=8,  # 单个 GPU 的 dataloader 的进程
        train_dataloader=dict(  # 训练过程 dataloader 的额外设置
            drop_last=True),  # 在训练过程中是否丢弃最后一个批次
        val_dataloader=dict(  # 验证过程 dataloader 的额外设置
            videos_per_gpu=1),  # 单个 GPU 的批大小
        test_dataloader=dict(  # 测试过程 dataloader 的额外设置
            videos_per_gpu=2),  # 单个 GPU 的批大小
        test=dict(  # 测试数据集的设置
            type=dataset_type,
            ann_file=ann_file_test,
            pipeline=test_pipeline,
            data_prefix=data_root_val),
        val=dict(  # 验证数据集的设置
            type=dataset_type,
            ann_file=ann_file_val,
            pipeline=val_pipeline,
            data_prefix=data_root_val),
        train=dict(  # 训练数据集的设置
            type=dataset_type,
            ann_file=ann_file_train,
            pipeline=train_pipeline,
            data_prefix=data_root))

    # 优化器设置
    optimizer = dict(
        # 构建优化器的设置，支持：
        # (1) 所有 PyTorch 原生的优化器，这些优化器的参数和 PyTorch 对应的一致；
        # (2) 自定义的优化器，这些优化器在 `constructor` 的基础上构建。
        # 更多细节可参考 "tutorials/5_new_modules.md" 部分
        type='Adam',  # 优化器类型, 参考 https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/optimizer/default_constructor.py#L13 for more details
        lr=0.001,  # 学习率, 参数的细节使用可参考 PyTorch 的对应文档
        weight_decay=0.0001)  # Adam 优化器的权重衰减
    optimizer_config = dict(  # 用于构建优化器钩子的设置
        grad_clip=None)  # 大部分的方法不使用梯度裁剪
    # 学习策略设置
    lr_config = dict(  # 用于注册学习率调整钩子的设置
        policy='step',  # 调整器策略, 支持 CosineAnnealing，Cyclic等方法。更多细节可参考 https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py#L9
        step=7)  # 学习率衰减步长

    total_epochs = 9  # 训练模型的总周期数
    checkpoint_config = dict(  # 模型权重文件钩子设置，更多细节可参考 https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint.py
        interval=1)  # 模型权重文件保存间隔
    evaluation = dict(  # 训练期间做验证的设置
        interval=1,  # 执行验证的间隔
        metrics=['AR@AN'])  # 验证方法
    log_config = dict(  # 注册日志钩子的设置
        interval=50,  # 打印日志间隔
        hooks=[  # 训练期间执行的钩子
            dict(type='TextLoggerHook'),  # 记录训练过程信息的日志
            # dict(type='TensorboardLoggerHook'),  # 同时支持 Tensorboard 日志
        ])

    # 运行设置
    dist_params = dict(backend='nccl')  # 建立分布式训练的设置（端口号，多 GPU 通信框架等）
    log_level = 'INFO'  # 日志等级
    work_dir = './work_dirs/bmn_400x100_2x8_9e_activitynet_feature/'  # 记录当前实验日志和模型权重文件的文件夹
    load_from = None  # 从给定路径加载模型作为预训练模型. 这个选项不会用于断点恢复训练
    resume_from = None  # 加载给定路径的模型权重文件作为断点续连的模型, 训练将从该时间点保存的周期点继续进行
    workflow = [('train', 1)]  # runner 的执行流. [('train', 1)] 代表只有一个执行流，并且这个名为 train 的执行流只执行一次
    output_config = dict(  # 时序检测器输出设置
        out=f'{work_dir}/results.json',  # 输出文件路径
        output_format='json')  # 输出文件格式
    ```

### 动作识别的配置文件系统

MMAction2 将模块化设计整合到配置文件系统中，以便执行各类不同实验。

- 以 TSN 为例

    为了帮助用户理解 MMAction2 的配置文件结构，以及动作识别系统中的一些模块，这里以 TSN 为例，给出其配置文件的注释。
    对于每个模块的详细用法以及对应参数的选择，请参照 [API 文档](https://mmaction2.readthedocs.io/en/latest/api.html)。

    ```python
    # 模型设置
    model = dict(  # 模型的配置
        type='Recognizer2D',  # 动作识别器的类型
        backbone=dict(  # Backbone 字典设置
            type='ResNet',  # Backbone 名
            pretrained='torchvision://resnet50',  # 预训练模型的 url 或文件位置
            depth=50,  # ResNet 模型深度
            norm_eval=False),  # 训练时是否设置 BN 层为验证模式
        cls_head=dict(  # 分类器字典设置
            type='TSNHead',  # 分类器名
            num_classes=400,  # 分类类别数量
            in_channels=2048,  # 分类器里输入通道数
            spatial_type='avg',  # 空间维度的池化种类
            consensus=dict(type='AvgConsensus', dim=1),  # consensus 模块设置
            dropout_ratio=0.4,  # dropout 层概率
            init_std=0.01), # 线性层初始化 std 值
            # 模型训练和测试的设置
        train_cfg=None,  # 训练 TSN 的超参配置
        test_cfg=dict(average_clips=None))  # 测试 TSN 的超参配置

    # 数据集设置
    dataset_type = 'RawframeDataset'  # 训练，验证，测试的数据集类型
    data_root = 'data/kinetics400/rawframes_train/'  # 训练集的根目录
    data_root_val = 'data/kinetics400/rawframes_val/'  # 验证集，测试集的根目录
    ann_file_train = 'data/kinetics400/kinetics400_train_list_rawframes.txt'  # 训练集的标注文件
    ann_file_val = 'data/kinetics400/kinetics400_val_list_rawframes.txt'  # 验证集的标注文件
    ann_file_test = 'data/kinetics400/kinetics400_val_list_rawframes.txt'  # 测试集的标注文件
    img_norm_cfg = dict(  # 图像正则化参数设置
        mean=[123.675, 116.28, 103.53],  # 图像正则化平均值
        std=[58.395, 57.12, 57.375],  # 图像正则化方差
        to_bgr=False)  # 是否将通道数从 RGB 转为 BGR

    train_pipeline = [  # 训练数据前处理流水线步骤组成的列表
        dict(  # SampleFrames 类的配置
            type='SampleFrames',  # 选定采样哪些视频帧
            clip_len=1,  # 每个输出视频片段的帧
            frame_interval=1,  # 所采相邻帧的时序间隔
            num_clips=3),  # 所采帧片段的数量
        dict(  # RawFrameDecode 类的配置
            type='RawFrameDecode'),  # 给定帧序列，加载对应帧，解码对应帧
        dict(  # Resize 类的配置
            type='Resize',  # 调整图片尺寸
            scale=(-1, 256)),  # 调整比例
        dict(  # MultiScaleCrop 类的配置
            type='MultiScaleCrop',  # 多尺寸裁剪，随机从一系列给定尺寸中选择一个比例尺寸进行裁剪
            input_size=224,  # 网络输入
            scales=(1, 0.875, 0.75, 0.66),  # 长宽比例选择范围
            random_crop=False,  # 是否进行随机裁剪
            max_wh_scale_gap=1),  # 长宽最大比例间隔
        dict(  # Resize 类的配置
            type='Resize',  # 调整图片尺寸
            scale=(224, 224),  # 调整比例
            keep_ratio=False),  # 是否保持长宽比
        dict(  # Flip 类的配置
            type='Flip',  # 图片翻转
            flip_ratio=0.5),  # 执行翻转几率
        dict(  # Normalize 类的配置
            type='Normalize',  # 图片正则化
            **img_norm_cfg),  # 图片正则化参数
        dict(  # FormatShape 类的配置
            type='FormatShape',  # 将图片格式转变为给定的输入格式
            input_format='NCHW'),  # 最终的图片组成格式
        dict(  # Collect 类的配置
            type='Collect',  # Collect 类决定哪些键会被传递到行为识别器中
            keys=['imgs', 'label'],  # 输入的键
            meta_keys=[]),  # 输入的元键
        dict(  # ToTensor 类的配置
            type='ToTensor',  # ToTensor 类将其他类型转化为 Tensor 类型
            keys=['imgs', 'label'])  # 将被从其他类型转化为 Tensor 类型的特征
    ]
    val_pipeline = [  # 验证数据前处理流水线步骤组成的列表
        dict(  # SampleFrames 类的配置
            type='SampleFrames',  # 选定采样哪些视频帧
            clip_len=1,  # 每个输出视频片段的帧
            frame_interval=1,  # 所采相邻帧的时序间隔
            num_clips=3,  # 所采帧片段的数量
            test_mode=True),  # 是否设置为测试模式采帧
        dict(  # RawFrameDecode 类的配置
            type='RawFrameDecode'),  # 给定帧序列，加载对应帧，解码对应帧
        dict(  # Resize 类的配置
            type='Resize',  # 调整图片尺寸
            scale=(-1, 256)),  # 调整比例
        dict(  # CenterCrop 类的配置
            type='CenterCrop',  # 中心裁剪
            crop_size=224),  # 裁剪部分的尺寸
        dict(  # Flip 类的配置
            type='Flip',  # 图片翻转
            flip_ratio=0),  # 翻转几率
        dict(  # Normalize 类的配置
            type='Normalize',  # 图片正则化
            **img_norm_cfg),  # 图片正则化参数
        dict(  # FormatShape 类的配置
            type='FormatShape',  # 将图片格式转变为给定的输入格式
            input_format='NCHW'),  # 最终的图片组成格式
        dict(  # Collect 类的配置
            type='Collect',  # Collect 类决定哪些键会被传递到行为识别器中
            keys=['imgs', 'label'],  # 输入的键
            meta_keys=[]),  # 输入的元键
        dict(  # ToTensor 类的配置
            type='ToTensor',  # ToTensor 类将其他类型转化为 Tensor 类型
            keys=['imgs'])  # 将被从其他类型转化为 Tensor 类型的特征
    ]
    test_pipeline = [  # 测试数据前处理流水线步骤组成的列表
        dict(  # SampleFrames 类的配置
            type='SampleFrames',  # 选定采样哪些视频帧
            clip_len=1,  # 每个输出视频片段的帧
            frame_interval=1,  # 所采相邻帧的时序间隔
            num_clips=25,  # 所采帧片段的数量
            test_mode=True),  # 是否设置为测试模式采帧
        dict(  # RawFrameDecode 类的配置
            type='RawFrameDecode'),  # 给定帧序列，加载对应帧，解码对应帧
        dict(  # Resize 类的配置
            type='Resize',  # 调整图片尺寸
            scale=(-1, 256)),  # 调整比例
        dict(  # TenCrop 类的配置
            type='TenCrop',  # 裁剪 10 个区域
            crop_size=224),  # 裁剪部分的尺寸
        dict(  # Flip 类的配置
            type='Flip',  # 图片翻转
            flip_ratio=0),  # 执行翻转几率
        dict(  # Normalize 类的配置
            type='Normalize',  # 图片正则化
            **img_norm_cfg),  # 图片正则化参数
        dict(  # FormatShape 类的配置
            type='FormatShape',  # 将图片格式转变为给定的输入格式
            input_format='NCHW'),  # 最终的图片组成格式
        dict(  # Collect 类的配置
            type='Collect',  # Collect 类决定哪些键会被传递到行为识别器中
            keys=['imgs', 'label'],  # 输入的键
            meta_keys=[]),  # 输入的元键
        dict(  # ToTensor 类的配置
            type='ToTensor',  # ToTensor 类将其他类型转化为 Tensor 类型
            keys=['imgs'])  # 将被从其他类型转化为 Tensor 类型的特征
    ]
    data = dict(  # 数据的配置
        videos_per_gpu=32,  # 单个 GPU 的批大小
        workers_per_gpu=2,  # 单个 GPU 的 dataloader 的进程
        train_dataloader=dict(  # 训练过程 dataloader 的额外设置
            drop_last=True),  # 在训练过程中是否丢弃最后一个批次
        val_dataloader=dict(  # 验证过程 dataloader 的额外设置
            videos_per_gpu=1),  # 单个 GPU 的批大小
        test_dataloader=dict(  # 测试过程 dataloader 的额外设置
            videos_per_gpu=2),  # 单个 GPU 的批大小
        train=dict(  # 训练数据集的设置
            type=dataset_type,
            ann_file=ann_file_train,
            data_prefix=data_root,
            pipeline=train_pipeline),
        val=dict(  # 验证数据集的设置
            type=dataset_type,
            ann_file=ann_file_val,
            data_prefix=data_root_val,
            pipeline=val_pipeline),
        test=dict(  # 测试数据集的设置
            type=dataset_type,
            ann_file=ann_file_test,
            data_prefix=data_root_val,
            pipeline=test_pipeline))
    # 优化器设置
    optimizer = dict(
        # 构建优化器的设置，支持：
        # (1) 所有 PyTorch 原生的优化器，这些优化器的参数和 PyTorch 对应的一致；
        # (2) 自定义的优化器，这些优化器在 `constructor` 的基础上构建。
        # 更多细节可参考 "tutorials/5_new_modules.md" 部分
        type='SGD',  # 优化器类型, 参考 https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/optimizer/default_constructor.py#L13
        lr=0.01,  # 学习率, 参数的细节使用可参考 PyTorch 的对应文档
        momentum=0.9,  # 动量大小
        weight_decay=0.0001)  # SGD 优化器权重衰减
    optimizer_config = dict(  # 用于构建优化器钩子的设置
        grad_clip=dict(max_norm=40, norm_type=2))  # 使用梯度裁剪
    # 学习策略设置
    lr_config = dict(  # 用于注册学习率调整钩子的设置
        policy='step',  # 调整器策略, 支持 CosineAnnealing，Cyclic等方法。更多细节可参考 https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py#L9
        step=[40, 80])  # 学习率衰减步长
    total_epochs = 100  # 训练模型的总周期数
    checkpoint_config = dict(  # 模型权重钩子设置，更多细节可参考 https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint.py
        interval=5)  # 模型权重文件保存间隔
    evaluation = dict(  # 训练期间做验证的设置
        interval=5,  # 执行验证的间隔
        metrics=['top_k_accuracy', 'mean_class_accuracy'],  # 验证方法
        save_best='top_k_accuracy')  # 设置 `top_k_accuracy` 作为指示器，用于存储最好的模型权重文件
    log_config = dict(  # 注册日志钩子的设置
        interval=20,  # 打印日志间隔
        hooks=[  # 训练期间执行的钩子
            dict(type='TextLoggerHook'),  # 记录训练过程信息的日志
            # dict(type='TensorboardLoggerHook'),  # 同时支持 Tensorboard 日志
        ])

    # 运行设置
    dist_params = dict(backend='nccl')  # 建立分布式训练的设置，其中端口号也可以设置
    log_level = 'INFO'  # 日志等级
    work_dir = './work_dirs/tsn_r50_1x1x3_100e_kinetics400_rgb/'  # 记录当前实验日志和模型权重文件的文件夹
    load_from = None  # 从给定路径加载模型作为预训练模型. 这个选项不会用于断点恢复训练
    resume_from = None  # 加载给定路径的模型权重文件作为断点续连的模型, 训练将从该时间点保存的周期点继续进行
    workflow = [('train', 1)]  # runner 的执行流. [('train', 1)] 代表只有一个执行流，并且这个名为 train 的执行流只执行一次

    ```

### 时空动作检测的配置文件系统

MMAction2 将模块化设计整合到配置文件系统中，以便于执行各种不同的实验。

- 以 FastRCNN 为例

    为了帮助用户理解 MMAction2 的完整配置文件结构，以及时空检测系统中的一些模块，这里以 FastRCNN 为例，给出其配置文件的注释。
    对于每个模块的详细用法以及对应参数的选择，请参照 [API 文档](https://mmaction2.readthedocs.io/en/latest/api.html)。

    ```python
    # 模型设置
    model = dict(  # 模型的配置
        type='FastRCNN',  # 时空检测器类型
        backbone=dict(  # Backbone 字典设置
            type='ResNet3dSlowOnly',  # Backbone 名
            depth=50, # ResNet 模型深度
            pretrained=None,   # 预训练模型的 url 或文件位置
            pretrained2d=False, # 预训练模型是否为 2D 模型
            lateral=False,  # backbone 是否有侧连接
            num_stages=4, # ResNet 模型阶数
            conv1_kernel=(1, 7, 7), # Conv1 卷积核尺寸
            conv1_stride_t=1, # Conv1 时序步长
            pool1_stride_t=1, # Pool1 时序步长
            spatial_strides=(1, 2, 2, 1)),  # 每个 ResNet 阶的空间步长
        roi_head=dict(  # roi_head 字典设置
            type='AVARoIHead',  # roi_head 名
            bbox_roi_extractor=dict(  # bbox_roi_extractor 字典设置
                type='SingleRoIExtractor3D',  # bbox_roi_extractor 名
                roi_layer_type='RoIAlign',  # RoI op 类型
                output_size=8,  # RoI op 输出特征尺寸
                with_temporal_pool=True), # 时序维度是否要经过池化
            bbox_head=dict( # bbox_head 字典设置
                type='BBoxHeadAVA', # bbox_head 名
                in_channels=2048, # 输入特征通道数
                num_classes=81, # 动作类别数 + 1（背景）
                multilabel=True,  # 数据集是否多标签
                dropout_ratio=0.5)),  # dropout 比率
        # 模型训练和测试的设置
        train_cfg=dict(  # 训练 FastRCNN 的超参配置
            rcnn=dict(  # rcnn 训练字典设置
                assigner=dict(  # assigner 字典设置
                    type='MaxIoUAssignerAVA', # assigner 名
                    pos_iou_thr=0.9,  # 正样本 IoU 阈值, > pos_iou_thr -> positive
                    neg_iou_thr=0.9,  # 负样本 IoU 阈值, < neg_iou_thr -> negative
                    min_pos_iou=0.9), # 正样本最小可接受 IoU
                sampler=dict( # sample 字典设置
                    type='RandomSampler', # sampler 名
                    num=32, # sampler 批大小
                    pos_fraction=1, # sampler 正样本边界框比率
                    neg_pos_ub=-1,  # 负样本数转正样本数的比率上界
                    add_gt_as_proposals=True), # 是否添加 ground truth 为候选
                pos_weight=1.0, # 正样本 loss 权重
                debug=False)), # 是否为 debug 模式
        test_cfg=dict( # 测试 FastRCNN 的超参设置
            rcnn=dict(  # rcnn 测试字典设置
                action_thr=0.002))) # 某行为的阈值

    # 数据集设置
    dataset_type = 'AVADataset' # 训练，验证，测试的数据集类型
    data_root = 'data/ava/rawframes'  # 训练集的根目录
    anno_root = 'data/ava/annotations'  # 标注文件目录

    ann_file_train = f'{anno_root}/ava_train_v2.1.csv'  # 训练集的标注文件
    ann_file_val = f'{anno_root}/ava_val_v2.1.csv'  # 验证集的标注文件

    exclude_file_train = f'{anno_root}/ava_train_excluded_timestamps_v2.1.csv'  # 训练除外数据集文件路径
    exclude_file_val = f'{anno_root}/ava_val_excluded_timestamps_v2.1.csv'  # 验证除外数据集文件路径

    label_file = f'{anno_root}/ava_action_list_v2.1_for_activitynet_2018.pbtxt'  # 标签文件路径

    proposal_file_train = f'{anno_root}/ava_dense_proposals_train.FAIR.recall_93.9.pkl'  # 训练样本检测候选框的文件路径
    proposal_file_val = f'{anno_root}/ava_dense_proposals_val.FAIR.recall_93.9.pkl'  # 验证样本检测候选框的文件路径

    img_norm_cfg = dict(  # 图像正则化参数设置
        mean=[123.675, 116.28, 103.53], # 图像正则化平均值
        std=[58.395, 57.12, 57.375],   # 图像正则化方差
        to_bgr=False) # 是否将通道数从 RGB 转为 BGR

    train_pipeline = [  # 训练数据前处理流水线步骤组成的列表
        dict(  # SampleFrames 类的配置
            type='AVASampleFrames',  # 选定采样哪些视频帧
            clip_len=4,  # 每个输出视频片段的帧
            frame_interval=16), # 所采相邻帧的时序间隔
        dict(  # RawFrameDecode 类的配置
            type='RawFrameDecode'),  # 给定帧序列，加载对应帧，解码对应帧
        dict(  # RandomRescale 类的配置
            type='RandomRescale',   # 给定一个范围，进行随机短边缩放
            scale_range=(256, 320)),   # RandomRescale 的短边缩放范围
        dict(  # RandomCrop 类的配置
            type='RandomCrop',   # 给定一个尺寸进行随机裁剪
            size=256),   # 裁剪尺寸
        dict(  # Flip 类的配置
            type='Flip',  # 图片翻转
            flip_ratio=0.5),  # 执行翻转几率
        dict(  # Normalize 类的配置
            type='Normalize',  # 图片正则化
            **img_norm_cfg),  # 图片正则化参数
        dict(  # FormatShape 类的配置
            type='FormatShape',  # 将图片格式转变为给定的输入格式
            input_format='NCTHW',  # 最终的图片组成格式
            collapse=True),   # 去掉 N 梯度当 N == 1
        dict(  # Rename 类的配置
            type='Rename',  # 重命名 key 名
            mapping=dict(imgs='img')),  # 改名映射字典
        dict(  # ToTensor 类的配置
            type='ToTensor',  # ToTensor 类将其他类型转化为 Tensor 类型
            keys=['img', 'proposals', 'gt_bboxes', 'gt_labels']),  # 将被从其他类型转化为 Tensor 类型的特征
        dict(  # ToDataContainer 类的配置
            type='ToDataContainer',  # 将一些信息转入到 ToDataContainer 中
            fields=[   # 转化为 Datacontainer 的域
                dict(   # 域字典
                    key=['proposals', 'gt_bboxes', 'gt_labels'],  # 将转化为 DataContainer 的键
                    stack=False)]),  # 是否要堆列这些 tensor
        dict(  # Collect 类的配置
            type='Collect',  # Collect 类决定哪些键会被传递到时空检测器中
            keys=['img', 'proposals', 'gt_bboxes', 'gt_labels'],  # 输入的键
            meta_keys=['scores', 'entity_ids']),  # 输入的元键
    ]

    val_pipeline = [  # 验证数据前处理流水线步骤组成的列表
        dict(  # SampleFrames 类的配置
            type='AVASampleFrames',  # 选定采样哪些视频帧
            clip_len=4,  # 每个输出视频片段的帧
            frame_interval=16),  # 所采相邻帧的时序间隔
        dict(  # RawFrameDecode 类的配置
            type='RawFrameDecode'),  # 给定帧序列，加载对应帧，解码对应帧
        dict(  # Resize 类的配置
            type='Resize',  # 调整图片尺寸
            scale=(-1, 256)),  # 调整比例
        dict(  # Normalize 类的配置
            type='Normalize',  # 图片正则化
            **img_norm_cfg),  # 图片正则化参数
        dict(  # FormatShape 类的配置
            type='FormatShape',  # 将图片格式转变为给定的输入格式
            input_format='NCTHW',  # 最终的图片组成格式
            collapse=True),   # 去掉 N 梯度当 N == 1
        dict(  # Rename 类的配置
            type='Rename',  # 重命名 key 名
            mapping=dict(imgs='img')),  # 改名映射字典
        dict(  # ToTensor 类的配置
            type='ToTensor',  # ToTensor 类将其他类型转化为 Tensor 类型
            keys=['img', 'proposals']),  # 将被从其他类型转化为 Tensor 类型的特征
        dict(  # ToDataContainer 类的配置
            type='ToDataContainer',  # 将一些信息转入到 ToDataContainer 中
            fields=[   # 转化为 Datacontainer 的域
                dict(   # 域字典
                    key=['proposals'],  # 将转化为 DataContainer 的键
                    stack=False)]),  # 是否要堆列这些 tensor
        dict(  # Collect 类的配置
            type='Collect',  # Collect 类决定哪些键会被传递到时空检测器中
            keys=['img', 'proposals'],  # 输入的键
            meta_keys=['scores', 'entity_ids'],  # 输入的元键
            nested=True)  # 是否将数据包装为嵌套列表
    ]

    data = dict(  # 数据的配置
        videos_per_gpu=16,  # 单个 GPU 的批大小
        workers_per_gpu=2,  # 单个 GPU 的 dataloader 的进程
        val_dataloader=dict(   # 验证过程 dataloader 的额外设置
            videos_per_gpu=1),  # 单个 GPU 的批大小
        train=dict(   # 训练数据集的设置
            type=dataset_type,
            ann_file=ann_file_train,
            exclude_file=exclude_file_train,
            pipeline=train_pipeline,
            label_file=label_file,
            proposal_file=proposal_file_train,
            person_det_score_thr=0.9,
            data_prefix=data_root),
        val=dict(     # 验证数据集的设置
            type=dataset_type,
            ann_file=ann_file_val,
            exclude_file=exclude_file_val,
            pipeline=val_pipeline,
            label_file=label_file,
            proposal_file=proposal_file_val,
            person_det_score_thr=0.9,
            data_prefix=data_root))
    data['test'] = data['val']    # 将验证数据集设置复制到测试数据集设置

    # 优化器设置
    optimizer = dict(
        # 构建优化器的设置，支持：
        # (1) 所有 PyTorch 原生的优化器，这些优化器的参数和 PyTorch 对应的一致；
        # (2) 自定义的优化器，这些优化器在 `constructor` 的基础上构建。
        # 更多细节可参考 "tutorials/5_new_modules.md" 部分
        type='SGD',  # 优化器类型, 参考 https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/optimizer/default_constructor.py#L13
        lr=0.2,  # 学习率, 参数的细节使用可参考 PyTorch 的对应文档
        momentum=0.9,  # 动量大小
        weight_decay=0.00001)  # SGD 优化器权重衰减

    optimizer_config = dict(  # 用于构建优化器钩子的设置
        grad_clip=dict(max_norm=40, norm_type=2))   # 使用梯度裁剪

    lr_config = dict(  # 用于注册学习率调整钩子的设置
        policy='step',  # 调整器策略, 支持 CosineAnnealing，Cyclic等方法。更多细节可参考 https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py#L9
        step=[40, 80],  # 学习率衰减步长
        warmup='linear',  # Warmup 策略
        warmup_by_epoch=True,  # Warmup 单位为 epoch 还是 iteration
        warmup_iters=5,   # warmup 数
        warmup_ratio=0.1)   # 初始学习率为 warmup_ratio * lr

    total_epochs = 20  # 训练模型的总周期数
    checkpoint_config = dict(  # 模型权重文件钩子设置，更多细节可参考 https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint.py
        interval=1)   # 模型权重文件保存间隔
    workflow = [('train', 1)]   # runner 的执行流. [('train', 1)] 代表只有一个执行流，并且这个名为 train 的执行流只执行一次
    evaluation = dict(  # 训练期间做验证的设置
        interval=1, save_best='mAP@0.5IOU')  # 执行验证的间隔，以及设置 `mAP@0.5IOU` 作为指示器，用于存储最好的模型权重文件
    log_config = dict(  # 注册日志钩子的设置
        interval=20,  # 打印日志间隔
        hooks=[  # 训练期间执行的钩子
            dict(type='TextLoggerHook'),  # 记录训练过程信息的日志
        ])

    # 运行设置
    dist_params = dict(backend='nccl')  # 建立分布式训练的设置，其中端口号也可以设置
    log_level = 'INFO'  # 日志等级
    work_dir = ('./work_dirs/ava/'  # 记录当前实验日志和模型权重文件的文件夹
                'slowonly_kinetics_pretrained_r50_4x16x1_20e_ava_rgb')
    load_from = ('https://download.openmmlab.com/mmaction/recognition/slowonly/'  # 从给定路径加载模型作为预训练模型. 这个选项不会用于断点恢复训练
                 'slowonly_r50_4x16x1_256e_kinetics400_rgb/'
                 'slowonly_r50_4x16x1_256e_kinetics400_rgb_20200704-a69556c6.pth')
    resume_from = None  # 加载给定路径的模型权重文件作为断点续连的模型, 训练将从该时间点保存的周期点继续进行
    ```

## 常见问题

### 配置文件中的中间变量

配置文件中会用到一些中间变量，如 `train_pipeline`/`val_pipeline`/`test_pipeline`, `ann_file_train`/`ann_file_val`/`ann_file_test`, `img_norm_cfg` 等。

例如，首先定义中间变量 `train_pipeline`/`val_pipeline`/`test_pipeline`，再将上述变量传递到 `data`。因此，`train_pipeline`/`val_pipeline`/`test_pipeline` 为中间变量

这里也定义了 `ann_file_train`/`ann_file_val`/`ann_file_test` 和 `data_root`/`data_root_val` 为数据处理流程提供一些基本信息。

此外，使用 `img_norm_cfg` 作为中间变量，构建一些数组增强组件。

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
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]

data = dict(
    videos_per_gpu=8,
    workers_per_gpu=2,
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
