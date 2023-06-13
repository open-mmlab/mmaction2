# 教程 1：如何编写配置文件

MMAction2 使用 python 文件作为配置文件。其配置文件系统的设计将模块化与继承整合进来，方便用户进行各种实验。
MMAction2 提供的所有配置文件都放置在 `$MMAction2/configs` 文件夹下，用户可以通过运行命令
`python tools/analysis_tools/print_config.py /PATH/TO/CONFIG` 来查看完整的配置信息，从而方便检查所对应的配置文件。

<!-- TOC -->

- [通过命令行参数修改配置信息](#通过命令行参数修改配置信息)
- [配置文件结构](#配置文件结构)
- [配置文件命名规则](#配置文件命名规则)
  - [动作识别的配置文件系统](#动作识别的配置文件系统)
  - [时空动作检测的配置文件系统](#时空动作检测的配置文件系统)
  - [时序动作检测的配置文件系统](#时序动作检测的配置文件系统)

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

  当配置文件中需要更新的是一个列表或者元组，例如，配置文件通常会设置 `model.data_preprocessor.mean=[123.675, 116.28, 103.53]`，用户如果想更改，
  需要指定 `--cfg-options model.data_preprocessor.mean="[128,128,128]"`。注意这里的引号 " 对于列表/元组数据类型的修改是必要的。

## 配置文件结构

在 `config/_base_` 文件夹下存在 3 种基本组件类型： 模型（model）, 训练策略（schedule）, 运行时的默认设置（default_runtime）。
许多方法都可以方便地通过组合这些组件进行实现，如 TSN，I3D，SlowOnly 等。
其中，通过 `_base_` 下组件来构建的配置被称为 _原始配置_（_primitive_）。

对于在同一文件夹下的所有配置文件，MMAction2 推荐只存在 **一个** 对应的 _原始配置_ 文件。
所有其他的配置文件都应该继承 _原始配置_ 文件，这样就能保证配置文件的最大继承深度为 3。

为了方便理解，MMAction2 推荐用户继承现有方法的配置文件。
例如，如需修改 TSN 的配置文件，用户应先通过 `_base_ = '../tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py'` 继承 TSN 配置文件的基本结构，
并修改其中必要的内容以完成继承。

如果用户想实现一个独立于任何一个现有的方法结构的新方法，则可以在 `configs/TASK` 中建立新的文件夹。

更多详细内容，请参考 [mmengine](https://mmengine.readthedocs.io/en/latest/tutorials/config.html)。

## 配置文件命名规则

MMAction2 按照以下风格进行配置文件命名，代码库的贡献者需要遵循相同的命名规则。配置文件名分为几个部分。逻辑上，不同的部分用下划线 `'_'`连接，同一部分中的设置用破折号 `'-'`连接。

```
{algorithm info}_{module info}_{training info}_{data info}.py
```

其中，`{xxx}` 表示必要的命名域，`[yyy]` 表示可选的命名域。

- `{algorithm info}`:
  - `{model}`: 模型类型，如 `tsn`，`i3d`, `swin`, `vit` 等。
  - `[model setting]`: 一些模型上的特殊设置,如`base`, `p16`, `w877`等。
- `{module info}`:
  - `[pretained info]`: 预训练信息,如 `kinetics400-pretrained`， `in1k-pre`等.
  - `{backbone}`: 主干网络类型和预训练信息，如 `r50`（ResNet-50）等。
  - `[backbone setting]`: 对于一些骨干网络的特殊设置，如`nl-dot-product`, `bnfrozen`, `nopool`等。
- `{training info}`:
  - `{gpu x batch_per_gpu]}`: GPU 数量以及每个 GPU 上的采样。
  - `{pipeline setting}`: 采帧数据格式，形如 `dense`, `{clip_len}x{frame_interval}x{num_clips}`, `u48`等。
  - `{schedule}`: 训练策略设置，如 `20e` 表示 20 个周期（epoch）。
- `{data info}`:
  - `{dataset}`:数据集名，如 `kinetics400`，`mmit`等。
  - `{modality}`: 帧的模态，如 `rgb`, `flow`, `keypoint-2d`等。

### 动作识别的配置文件系统

MMAction2 将模块化设计整合到配置文件系统中，以便执行各类不同实验。

- 以 TSN 为例

  为了帮助用户理解 MMAction2 的配置文件结构，以及动作识别系统中的一些模块，这里以 TSN 为例，给出其配置文件的注释。
  对于每个模块的详细用法以及对应参数的选择，请参照 API 文档。

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
          init_std=0.01,  # 线性层初始化 std 值
          average_clips='prob'),  # 平均多个 clip 结果的方法
      data_preprocessor=dict(  # 数据预处理器的字典设置
          type='ActionDataPreprocessor',  # 数据预处理器名
          mean=[123.675, 116.28, 103.53],  # 不同通道归一化所用的平均值
          std=[58.395, 57.12, 57.375],  # 不同通道归一化所用的方差
          format_shape='NCHW'),  # 最终图像形状格式
      # 模型训练和测试的设置
      train_cfg=None,  # 训练 TSN 的超参配置
      test_cfg=None)  # 测试 TSN 的超参配置

  # 数据集设置
  dataset_type = 'RawframeDataset'  # 训练，验证，测试的数据集类型
  data_root = 'data/kinetics400/rawframes_train/'  # 训练集的根目录
  data_root_val = 'data/kinetics400/rawframes_val/'  # 验证集，测试集的根目录
  ann_file_train = 'data/kinetics400/kinetics400_train_list_rawframes.txt'  # 训练集的标注文件
  ann_file_val = 'data/kinetics400/kinetics400_val_list_rawframes.txt'  # 验证集的标注文件
  ann_file_test = 'data/kinetics400/kinetics400_val_list_rawframes.txt'  # 测试集的标注文件

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
      dict(  # FormatShape 类的配置
          type='FormatShape',  # 将图片格式转变为给定的输入格式
          input_format='NCHW'),  # 最终的图片组成格式
      dict(  # PackActionInputs 类的配置
          type='PackActionInputs')  # 将输入数据打包
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
      dict(  # FormatShape 类的配置
          type='FormatShape',  # 将图片格式转变为给定的输入格式
          input_format='NCHW'),  # 最终的图片组成格式
      dict(  # PackActionInputs 类的配置
          type='PackActionInputs')  # 将输入数据打包
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
      dict(  # FormatShape 类的配置
          type='FormatShape',  # 将图片格式转变为给定的输入格式
          input_format='NCHW'),  # 最终的图片组成格式
      dict(  # PackActionInputs 类的配置
          type='PackActionInputs')  # 将输入数据打包
  ]

  train_dataloader = dict(  # 训练过程 dataloader 的配置
      batch_size=32,  # 训练过程单个 GPU 的批大小
      num_workers=8,  # 训练过程单个 GPU 的 数据预取的进程数
      persistent_workers=True,  # 保持`Dataset` 实例
      sampler=dict(type='DefaultSampler', shuffle=True),
      dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(video=data_root),
        pipeline=train_pipeline))
  val_dataloader = dict(  # 验证过程 dataloader 的配置
      batch_size=1,  # 验证过程单个 GPU 的批大小
      num_workers=8,  # 验证过程单个 GPU 的 数据预取的进程
      persistent_workers=True,  # 保持`Dataset` 实例
      sampler=dict(type='DefaultSampler', shuffle=False),
      dataset=dict(
          type=dataset_type,
          ann_file=ann_file_val,
          data_prefix=dict(video=data_root_val),
          pipeline=val_pipeline,
          test_mode=True))
  test_dataloader = dict(  # 测试过程 dataloader 的配置
      batch_size=32,  # 测试过程单个 GPU 的批大小
      num_workers=8,  # 测试过程单个 GPU 的 数据预取的进程
      persistent_workers=True,  # 保持`Dataset` 实例
      sampler=dict(type='DefaultSampler', shuffle=False),
      dataset=dict(
          type=dataset_type,
          ann_file=ann_file_val,
          data_prefix=dict(video=data_root_val),
          pipeline=test_pipeline,
          test_mode=True))

  # 评测器设置
  val_evaluator = dict(type='AccMetric')  # 用于计算验证指标的评测对象
  test_evaluator = dict(type='AccMetric')  # 用于计算测试指标的评测对象

  train_cfg = dict(  # 训练循环的配置
      type='EpochBasedTrainLoop',  # 训练循环的名称
      max_epochs=100,  # 整体循环次数
      val_begin=1,  # 开始验证的轮次
      val_interval=1)  # 执行验证的间隔
  val_cfg = dict(  # 验证循环的配置
      type='ValLoop')  # 验证循环的名称
  test_cfg = dict( # 测试循环的配置
      type='TestLoop')  # 测试循环的名称

  # 学习策略设置
  param_scheduler = [  # 用于更新优化器参数的参数调度程序，支持字典或列表
      dict(type='MultiStepLR',  # 当轮次数达到阈值，学习率衰减
          begin=0,  # 开始更新学习率的步长
          end=100,  # 停止更新学习率的步长
          by_epoch=True,  # 学习率是否按轮次更新
          milestones=[40, 80],  # 学习率衰减阈值
          gamma=0.1)]  # 学习率衰减的乘数因子

  # 优化器设置
  optim_wrapper = dict(  # 优化器钩子的配置
      type='OptimWrapper',  #  优化器封装的名称, 切换到 AmpOptimWrapper 可以实现混合精度训练
      optimizer=dict(  # 优化器配置。 支持各种在pytorch上的优化器。 参考 https://pytorch.org/docs/stable/optim.html#algorithms
          type='SGD',  # 优化器名称
          lr=0.01,  # 学习率
          momentum=0.9,  # 动量大小
          weight_decay=0.0001)  # SGD 优化器权重衰减
      clip_grad=dict(max_norm=40, norm_type=2))  # 梯度裁剪的配置

  # 运行设置
  default_scope = 'mmaction'  # 查找模块的默认注册表范围。 参考 https://mmengine.readthedocs.io/en/latest/tutorials/registry.html
  default_hooks = dict(  # 执行默认操作的钩子，如更新模型参数和保存checkpoints。
      runtime_info=dict(type='RuntimeInfoHook'),  # 将运行信息更新到消息中心的钩子。
      timer=dict(type='IterTimerHook'),  # 记录迭代期间花费时间的日志。
      logger=dict(
          type='LoggerHook',  # 记录训练/验证/测试阶段记录日志。
          interval=20,  # 打印日志间隔
          ignore_last=False), # 忽略每个轮次中最后一次迭代的日志
      param_scheduler=dict(type='ParamSchedulerHook'),  # 更新优化器中一些超参数的钩子
      checkpoint=dict(
          type='CheckpointHook',  # 定期保存检查点的钩子
          interval=3,  # 保存周期
          save_best='auto',  # 在评估期间测量最佳检查点的指标
          max_keep_ckpts=3),  # 要保留的最大检查点
      sampler_seed=dict(type='DistSamplerSeedHook'),  # 分布式训练的数据加载采样器
      sync_buffers=dict(type='SyncBuffersHook'))  # 在每个轮次结束时同步模型缓冲区
  env_cfg = dict(  # 环境设置
      cudnn_benchmark=False,  # 是否启用cudnn基准
      mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0), # 设置多线程处理的参数
      dist_cfg=dict(backend='nccl')) # 设置分布式环境的参数，也可以设置端口

  log_processor = dict(
      type='LogProcessor',  # 用于格式化日志信息的日志处理器
      window_size=20,  # 默认平滑间隔
      by_epoch=True)  # 是否以epoch类型格式化日志
  vis_backends = [  # 可视化后端列表
      dict(type='LocalVisBackend')]  # 本地可视化后端
  visualizer = dict(  # 可视化工具的配置
      type='ActionVisualizer',  # 可视化工具的名称
      vis_backends=vis_backends)
  log_level = 'INFO'  # 日志记录级别
  load_from = None  # 从给定路径加载模型checkpoint作为预训练模型。这不会恢复训练。
  resume = False  # 是否从`load_from`中定义的checkpoint恢复。如果“load_from”为“None”，它将恢复“work_dir”中的最新的checkpoint。
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
      _scope_='mmdet',  # 当前配置的范围
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
      data_preprocessor=dict(  # 数据预处理器的字典
          type='ActionDataPreprocessor',  # 数据预处理器的名称
          mean=[123.675, 116.28, 103.53],  # 不同通道归一化的均值
          std=[58.395, 57.12, 57.375],  # 不同通道归一化的方差
          format_shape='NCHW')),  # 最终图像形状
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
              pos_weight=1.0)), # 正样本 loss 权重
      test_cfg=dict( # 测试 FastRCNN 的超参设置
          rcnn=dict(rcnn=None))  # rcnn 测试字典设置

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
      dict(  # FormatShape 类的配置
          type='FormatShape',  # 将图片格式转变为给定的输入格式
          input_format='NCTHW',  # 最终的图片组成格式
          collapse=True),   # 去掉 N 梯度当 N == 1
      dict(type='PackActionInputs')# 打包输入数据
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
      dict(  # FormatShape 类的配置
          type='FormatShape',  # 将图片格式转变为给定的输入格式
          input_format='NCTHW',  # 最终的图片组成格式
          collapse=True),   # 去掉 N 梯度当 N == 1
      dict(type='PackActionInputs') # 打包输入数据
  ]

  train_dataloader = dict(  # 训练过程 dataloader 的配置
      batch_size=32,  # 训练过程单个 GPU 的批大小
      num_workers=8,  # 训练过程单个 GPU 的 数据预取的进程
      persistent_workers=True,  # 如果为“True”，则数据加载器不会在轮次结束后关闭工作进程，这可以加快训练速度
      sampler=dict(
          type='DefaultSampler', # 支持分布式和非分布式的DefaultSampler
          shuffle=True), 随机打乱每个轮次的训练数据
      dataset=dict(  # 训练数据集的配置
          type=dataset_type,
          ann_file=ann_file_train,  # 标注文件的路径
          exclude_file=exclude_file_train,  # 不包括的标注文件路径
          label_file=label_file,  # 标签文件的路径
          data_prefix=dict(img=data_root),  # 帧路径的前缀
          proposal_file=proposal_file_train,  # 行人检测框的路径
          pipeline=train_pipeline))
  val_dataloader = dict(  # 验证过程 dataloader 的配置
      batch_size=1,  # 验证过程单个 GPU 的批大小
      num_workers=8,  # 验证过程单个 GPU 的 数据预取的进程
      persistent_workers=True,  # 保持`Dataset` 实例
      sampler=dict(
          type='DefaultSampler',
          shuffle=False),  # 在验证测试期间不打乱数据
      dataset=dict(  # 验证集的配置
          type=dataset_type,
          ann_file=ann_file_val,  # 标注文件的路径
          exclude_file=exclude_file_train,  # 不包括的标注文件路径
          label_file=label_file,  # 标签文件的路径
          data_prefix=dict(video=data_root_val),  # 帧路径的前缀
          proposal_file=proposal_file_val,  # # 行人检测框的路径
          pipeline=val_pipeline,
          test_mode=True))
  test_dataloader = val_dataloader  # 测试过程 dataloader 的配置


  # 评估器设置
  val_evaluator = dict(  # 验证评估器的配置
      type='AccMetric',
      ann_file=ann_file_val,
      label_file=label_file,
      exclude_file=exclude_file_val)
  test_evaluator = val_evaluator  # 测试评估器的配置

  train_cfg = dict(  # 训练循环的配置
      type='EpochBasedTrainLoop',  # 训练循环的名称
      max_epochs=20,  # 整体循环次数
      val_begin=1,  # 开始验证的轮次
      val_interval=1)  # 执行验证的间隔
  val_cfg = dict(  # 验证循环的配置
      type='ValLoop')  # 验证循环的名称
  test_cfg = dict( # 测试循环的配置
      type='TestLoop')  # 测试循环的名称

  # 学习策略设置
  param_scheduler = [  # 用于更新优化器参数的参数调度程序，支持字典或列表
      dict（type='LinearLR'，# 通过乘法因子线性衰减来降低各参数组的学习率
          start_factor=0.1，# 乘以第一个轮次的学习率的数值
          by_epoch=True，# 学习率是否按轮次更新
          begin=0，# 开始更新学习率的步长
          end=5），# 停止更新学习率的步长
      dict(type='MultiStepLR',  # 当轮次数达到阈值，学习率衰减
          begin=0,  # 开始更新学习率的步长
          end=20,  # 停止更新学习率的步长
          by_epoch=True,  # 学习率是否按轮次更新
          milestones=[10, 15],  # 学习率衰减阈值
          gamma=0.1)]  # 学习率衰减的乘数因子


  # 优化器设置
  optim_wrapper = dict(  # 优化器钩子的配置
      type='OptimWrapper',  #  优化器封装的名称, 切换到 AmpOptimWrapper 可以实现混合精度训练
      optimizer=dict(  # 优化器配置。 支持各种在pytorch上的优化器。 参考 https://pytorch.org/docs/stable/optim.html#algorithms
          type='SGD',  # 优化器名称
          lr=0.2,  # 学习率
          momentum=0.9,  # 动量大小
          weight_decay=0.0001)  # SGD 优化器权重衰减
      clip_grad=dict(max_norm=40, norm_type=2))  # 梯度裁剪的配置

  # 运行设置
  default_scope = 'mmaction'  # 查找模块的默认注册表范围。 参考 https://mmengine.readthedocs.io/en/latest/tutorials/registry.html
  default_hooks = dict(  # 执行默认操作的钩子，如更新模型参数和保存checkpoints。
      runtime_info=dict(type='RuntimeInfoHook'),  # 将运行信息更新到消息中心的钩子。
      timer=dict(type='IterTimerHook'),  # 记录迭代期间花费时间的日志。
      logger=dict(
          type='LoggerHook',  # 记录训练/验证/测试阶段记录日志。
          interval=20,  # 打印日志间隔
          ignore_last=False), # 忽略每个轮次中最后一次迭代的日志
      param_scheduler=dict(type='ParamSchedulerHook'),  # 更新优化器中一些超参数的钩子
      checkpoint=dict(
          type='CheckpointHook',  # 定期保存检查点的钩子
          interval=3,  # 保存周期
          save_best='auto',  # 在评估期间测量最佳检查点的指标
          max_keep_ckpts=3),  # 要保留的最大检查点
      sampler_seed=dict(type='DistSamplerSeedHook'),  # 分布式训练的数据加载采样器
      sync_buffers=dict(type='SyncBuffersHook'))  # 在每个轮次结束时同步模型缓冲区
  env_cfg = dict(  # 环境设置
      cudnn_benchmark=False,  # 是否启用cudnn基准
      mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0), # 设置多线程处理的参数
      dist_cfg=dict(backend='nccl')) # 设置分布式环境的参数，也可以设置端口

  log_processor = dict(
      type='LogProcessor',  # 用于格式化日志信息的日志处理器
      window_size=20,  # 默认平滑间隔
      by_epoch=True)  # 是否以epoch类型格式化日志
  vis_backends = [  # 可视化后端列表
      dict(type='LocalVisBackend')]  # 本地可视化后端
  visualizer = dict(  # 可视化工具的配置
      type='ActionVisualizer',  # 可视化工具的名称
      vis_backends=vis_backends)
  log_level = 'INFO'  # 日志记录级别
  load_from = ('https://download.openmmlab.com/mmaction/v1.0/recognition/slowonly/'
               'slowonly_imagenet-pretrained-r50_8xb16-4x16x1-steplr-150e_kinetics400-rgb/'
               'slowonly_imagenet-pretrained-r50_8xb16-4x16x1-steplr-150e_kinetics400-rgb_20220901-e7b65fad.pth')  # 从给定路径加载模型checkpoint作为预训练模型。这不会恢复训练。
  resume = False  # 是否从`load_from`中定义的checkpoint恢复。如果“load_from”为“None”，它将恢复“work_dir”中的最新的checkpoint。
  ```

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
      dict(
          type='PackLocalizationInputs',  # 时序数据打包
          keys=('gt_bbox'),  # 输入的键
          meta_keys=('video_name'))]  # 输入的元键
  val_pipeline = [  # 验证数据前处理流水线步骤组成的列表
      dict(type='LoadLocalizationFeature'),  # 加载时序动作检测特征
      dict(type='GenerateLocalizationLabels'),  # 生成时序动作检测标签
      dict(
          type='PackLocalizationInputs',  # 时序数据打包
          keys=('gt_bbox'),  # 输入的键
          meta_keys= ('video_name', 'duration_second', 'duration_frame',
                      'annotations', 'feature_frame'))],  # 输入的元键
  test_pipeline = [  # 测试数据前处理流水线步骤组成的列表
      dict(type='LoadLocalizationFeature'),  # 加载时序动作检测特征
      dict(
          type='PackLocalizationInputs',  # 时序数据打包
          keys=('gt_bbox'),  # 输入的键
          meta_keys= ('video_name', 'duration_second', 'duration_frame',
                      'annotations', 'feature_frame'))],  # 输入的元键
  train_dataloader = dict(  # 训练过程 dataloader 的配置
      batch_size=8,  # 训练过程单个 GPU 的批大小
      num_workers=8,  # 训练过程单个 GPU 的 数据预取的进程
      persistent_workers=True,  # 如果为“True”，则数据加载器不会在轮次结束后关闭工作进程，这可以加快训练速度
      sampler=dict(
          type='DefaultSampler', # 支持分布式和非分布式的DefaultSampler
          shuffle=True), 随机打乱每个轮次的训练数据
      dataset=dict(  # 训练数据集的配置
        type=dataset_type,
        ann_file=ann_file_train,  # 标签文件的路径
        exclude_file=exclude_file_train,  # 不包括的标签文件路径
        label_file=label_file,  # 标签文件的路径
        data_prefix=dict(video=data_root),
        data_prefix=dict(img=data_root),  # Prefix of frame path
        pipeline=train_pipeline))
  val_dataloader = dict(  # 验证过程 dataloader 的配置
      batch_size=1,  # 验证过程单个 GPU 的批大小
      num_workers=8,  # 验证过程单个 GPU 的 数据预取的进程
      persistent_workers=True,  # 保持`Dataset` 实例
      sampler=dict(
          type='DefaultSampler',
          shuffle=False),  # 在验证测试过程中不打乱数据
      dataset=dict(  # 验证数据集的配置
          type=dataset_type,
          ann_file=ann_file_val,  # 标注文件的路径
          data_prefix=dict(video=data_root_val),  # 视频路径的前缀
          pipeline=val_pipeline,
          test_mode=True))
  test_dataloader = dict(  # 测试过程 dataloader 的配置
      batch_size=1,  #测试过程单个 GPU 的批大小
      num_workers=8,  # 测试过程单个 GPU 的 数据预取的进程
      persistent_workers=True,  # 保持`Dataset` 实例
      sampler=dict(
          type='DefaultSampler',
          shuffle=False),  # 在验证测试过程中不打乱数据
      dataset=dict(  # 测试数据集的配置
          type=dataset_type,
          ann_file=ann_file_val,  # 标注文件的路径
          data_prefix=dict(video=data_root_val),  # 视频路径的前缀
          pipeline=test_pipeline,
          test_mode=True))


  # 评估器设置
  work_dir = './work_dirs/bmn_400x100_2x8_9e_activitynet_feature/'  # 用于保存当前试验的模型检查点和日志的目录
  val_evaluator = dict(  # 验证评估器的配置
    type='AccMetric',
    metric_type='AR@AN',
    dump_config=dict(  # 时序输出的配置
        out=f'{work_dir}/results.json',  # 输出文件的路径
        output_format='json'))  # 输出文件的文件格式
  test_evaluator = val_evaluator  # 测试评估器的配置

  max_epochs = 9  # Total epochs to train the model
  train_cfg = dict(  # 训练循环的配置
     type='EpochBasedTrainLoop',  # 训练循环的名称
     max_epochs=100,  # 整体循环次数
     val_begin=1,  # 开始验证的轮次
     val_interval=1)  # 执行验证的间隔
  val_cfg = dict(  # 验证循环的配置
     type='ValLoop')  # 验证循环的名称
  test_cfg = dict( # 测试循环的配置
     type='TestLoop')  # 测试循环的名称

  # 学习策略设置
  param_scheduler = [  # 用于更新优化器参数的参数调度程序，支持字典或列表
     dict(type='MultiStepLR',  # 当轮次数达到阈值，学习率衰减
     begin=0,  # 开始更新学习率的步长
     end=max_epochs,  # 停止更新学习率的步长
     by_epoch=True,  # 学习率是否按轮次更新
     milestones=[7, ],  # 学习率衰减阈值
     gamma=0.1)]  # 学习率衰减的乘数因子

  # 优化器设置
  optim_wrapper = dict(  # 优化器钩子的配置
    type='OptimWrapper',  #  优化器封装的名称, 切换到 AmpOptimWrapper 可以实现混合精度训练
    optimizer=dict(  # 优化器配置。 支持各种在pytorch上的优化器。 参考 https://pytorch.org/docs/stable/optim.html#algorithms
      type='Adam',  # 优化器名称
      lr=0.001,  # 学习率
      weight_decay=0.0001)  # 权重衰减
    clip_grad=dict(max_norm=40, norm_type=2))  # 梯度裁剪的配置

  # 运行设置
  default_scope = 'mmaction'  # 查找模块的默认注册表范围。 参考 https://mmengine.readthedocs.io/en/latest/tutorials/registry.html
  default_hooks = dict(  # 执行默认操作的钩子，如更新模型参数和保存checkpoints。
      runtime_info=dict(type='RuntimeInfoHook'),  # 将运行信息更新到消息中心的钩子。
      timer=dict(type='IterTimerHook'),  # 记录迭代期间花费时间的日志。
      logger=dict(
          type='LoggerHook',  # 记录训练/验证/测试阶段记录日志。
          interval=20,  # 打印日志间隔
          ignore_last=False), # 忽略每个轮次中最后一次迭代的日志
      param_scheduler=dict(type='ParamSchedulerHook'),  # 更新优化器中一些超参数的钩子
      checkpoint=dict(
          type='CheckpointHook',  # 定期保存检查点的钩子
          interval=3,  # 保存周期
          save_best='auto',  # 在评估期间测量最佳检查点的指标
          max_keep_ckpts=3),  # 要保留的最大检查点
      sampler_seed=dict(type='DistSamplerSeedHook'),  # 分布式训练的数据加载采样器
      sync_buffers=dict(type='SyncBuffersHook'))  # 在每个轮次结束时同步模型缓冲区
  env_cfg = dict(  # 环境设置
      cudnn_benchmark=False,  # 是否启用cudnn基准
      mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0), # 设置多线程处理的参数
      dist_cfg=dict(backend='nccl')) # 设置分布式环境的参数，也可以设置端口

  log_processor = dict(
      type='LogProcessor',  # 用于格式化日志信息的日志处理器
      window_size=20,  # 默认平滑间隔
      by_epoch=True)  # 是否以epoch类型格式化日志
  vis_backends = [  # 可视化后端列表
      dict(type='LocalVisBackend')]  # 本地可视化后端
  visualizer = dict(  # 可视化工具的配置
      type='ActionVisualizer',  # 可视化工具的名称
      vis_backends=vis_backends)
  log_level = 'INFO'  # 日志记录级别
  load_from = None  # 从给定路径加载模型checkpoint作为预训练模型。这不会恢复训练。
  resume = False  # 是否从`load_from`中定义的checkpoint恢复。如果“load_from”为“None”，它将恢复“work_dir”中的最新的checkpoint。
  ```
