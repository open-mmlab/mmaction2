# 学习配置文件

我们使用 Python 文件作为配置文件，将模块化和继承设计融入我们的配置系统中，这方便进行各种实验。
您可以在 `$MMAction2/configs` 目录下找到所有提供的配置文件。如果您想要查看配置文件，
您可以运行 `python tools/analysis_tools/print_config.py /PATH/TO/CONFIG` 来查看完整的配置文件。

<!-- TOC -->

- [学习配置文件](#学习配置文件)
  - [通过脚本参数修改配置](#通过脚本参数修改配置)
  - [配置文件结构](#配置文件结构)
  - [配置文件命名约定](#配置文件命名约定)
    - [动作识别的配置系统](#动作识别的配置系统)
    - [时空动作检测的配置系统](#时空动作检测的配置系统)
    - [动作定位的配置系统](#动作定位的配置系统)

<!-- TOC -->

## 通过脚本参数修改配置

在使用 `tools/train.py` 或 `tools/test.py` 提交作业时，您可以通过指定 `--cfg-options` 来原地修改配置。

- 更新字典的配置键。

  可以按照原始配置中字典键的顺序来指定配置选项。
  例如，`--cfg-options model.backbone.norm_eval=False` 将模型骨干中的所有 BN 模块更改为 `train` 模式。

- 更新配置列表中的键。

  一些配置字典在配置文件中以列表形式组成。例如，训练流程 `train_pipeline` 通常是一个列表，
  例如 `[dict(type='SampleFrames'), ...]`。如果您想要在流程中将 `'SampleFrames'` 更改为 `'DenseSampleFrames'`，
  您可以指定 `--cfg-options train_pipeline.0.type=DenseSampleFrames`。

- 更新列表/元组的值。

  如果要更新的值是列表或元组。例如，配置文件通常设置 `model.data_preprocessor.mean=[123.675, 116.28, 103.53]`。如果您想要
  更改此键，您可以指定 `--cfg-options model.data_preprocessor.mean="[128,128,128]"`。请注意，引号 " 是支持列表/元组数据类型的必需内容。

## 配置文件结构

`configs/_base_` 下有 3 种基本组件类型，即 models、schedules 和 default_runtime。
许多方法只需要一个模型、一个训练计划和一个默认运行时组件就可以轻松构建，如 TSN、I3D、SlowOnly 等。
由 `_base_` 组件组成的配置文件被称为 _primitive_。

对于同一文件夹下的所有配置文件，建议只有**一个** _primitive_ 配置文件。其他所有配置文件都应该继承自 _primitive_ 配置文件。这样，继承级别的最大值为 3。

为了方便理解，我们建议贡献者继承现有方法。
例如，如果基于 TSN 进行了一些修改，用户可以首先通过指定 `_base_ = ../tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py` 来继承基本的 TSN 结构，然后在配置文件中修改必要的字段。

如果您正在构建一个与任何现有方法的结构不共享的全新方法，可以在 `configs/TASK` 下创建一个文件夹。

请参考 [mmengine](https://mmengine.readthedocs.io/en/latest/tutorials/config.html) 获取详细文档。

## 配置文件命名约定

我们遵循以下样式来命名配置文件。建议贡献者遵循相同的样式。配置文件名分为几个部分，不同部分逻辑上用下划线 `'_'` 连接，同一部分的设置用破折号 `'-'` 连接。

```
{算法信息}_{模块信息}_{训练信息}_{数据信息}.py
```

`{xxx}` 是必填字段，`[yyy]` 是可选字段。

- `{算法信息}`:
  - `{模型}`: 模型类型，例如 `tsn`、`i3d`、`swin`、`vit` 等。
  - `[模型设置]`: 某些模型的特定设置，例如 `base`、`p16`、`w877` 等。
- `{模块信息}`:
  - `[预训练信息]`: 预训练信息，例如 `kinetics400-pretrained`、`in1k-pre` 等。
  - `{骨干网络}`: 骨干网络类型，例如 `r50`（ResNet-50）等。
  - `[骨干网络设置]`: 某些骨干网络的特定设置，例如 `nl-dot-product`、`bnfrozen`、`nopool` 等。
- `{训练信息}`:
  - `{gpu x batch_per_gpu]}`: GPU 和每个 GPU 上的样本数。
  - `{pipeline设置}`: 帧采样设置，例如 `dense`、`{clip_len}x{frame_interval}x{num_clips}`、`u48` 等。
  - `{schedule}`: 训练计划，例如 `coslr-20e`。
- `{数据信息}`:
  - `{数据集}`: 数据集名称，例如 `kinetics400`、`mmit` 等。
  - `{模态}`: 数据模态，例如 `rgb`、`flow`、`keypoint-2d` 等。

### 动作识别的配置系统

我们将模块化设计融入我们的配置系统中，
这方便进行各种实验。

- TSN 的示例

  为了帮助用户对完整的配置结构和动作识别系统中的模块有一个基本的了解，
  我们对 TSN 的配置进行简要注释如下。有关每个模块中每个参数的更详细用法和替代方法，请参阅 API 文档。

  ```python
  # 模型设置
  model = dict(  # 模型的配置
      type='Recognizer2D',  # 识别器的类名
      backbone=dict(  # 骨干网络的配置
          type='ResNet',  # 骨干网络的名称
          pretrained='torchvision://resnet50',  # 预训练模型的 URL/网站
          depth=50,  # ResNet 模型的深度
          norm_eval=False),  # 是否在训练时将 BN 层设置为评估模式
      cls_head=dict(  # 分类头的配置
          type='TSNHead',  # 分类头的名称
          num_classes=400,  # 要分类的类别数量。
          in_channels=2048,  # 分类头的输入通道数。
          spatial_type='avg',  # 空间维度池化的类型
          consensus=dict(type='AvgConsensus', dim=1),  # 一致性模块的配置
          dropout_ratio=0.4,  # dropout 层中的概率
          init_std=0.01, # 线性层初始化的标准差值
          average_clips='prob'),  # 平均多个剪辑结果的方法
      data_preprocessor=dict(  # 数据预处理器的配置
          type='ActionDataPreprocessor',  # 数据预处理器的名称
          mean=[123.675, 116.28, 103.53],  # 不同通道的均值用于归一化
          std=[58.395, 57.12, 57.375],  # 不同通道的标准差用于归一化
          format_shape='NCHW'),  # 最终图像形状的格式
      # 模型训练和测试设置
      train_cfg=None,  # TSN 的训练超参数的配置
      test_cfg=None)  # TSN 的测试超参数的配置

  # 数据集设置
  dataset_type = 'RawframeDataset'  # 用于训练、验证和测试的数据集类型
  data_root = 'data/kinetics400/rawframes_train/'  # 用于训练的数据的根路径
  data_root_val = 'data/kinetics400/rawframes_val/'  # 用于验证和测试的数据的根路径
  ann_file_train = 'data/kinetics400/kinetics400_train_list_rawframes.txt'  # 用于训练的注释文件的路径
  ann_file_val = 'data/kinetics400/kinetics400_val_list_rawframes.txt'  # 用于验证的注释文件的路径
  ann_file_test = 'data/kinetics400/kinetics400_val_list_rawframes.txt'  # 用于测试的注释文件的路径

  train_pipeline = [  # 训练数据处理流程
      dict(  # SampleFrames 的配置
          type='SampleFrames',  # 采样帧的流程，从视频中采样帧
          clip_len=1,  # 每个采样输出剪辑的帧数
          frame_interval=1,  # 相邻采样帧的时间间隔
          num_clips=3),  # 要采样的剪辑数
      dict(  # RawFrameDecode 的配置
          type='RawFrameDecode'),  # 加载和解码帧的流程，选择给定索引的原始帧
      dict(  # Resize 的配置
          type='Resize',  # 调整大小的流程
          scale=(-1, 256)),  # 要调整图像的比例
      dict(  # MultiScaleCrop 的配置
          type='MultiScaleCrop',  # 多尺度裁剪的流程，根据随机选择的尺度列表裁剪图像
          input_size=224,  # 网络的输入大小
          scales=(1, 0.875, 0.75, 0.66),  # 要选择的宽度和高度的尺度
          random_crop=False,  # 是否随机采样裁剪框
          max_wh_scale_gap=1),  # 宽度和高度尺度级别的最大差距
      dict(  # Resize 的配置
          type='Resize',  # 调整大小的流程
          scale=(224, 224),  # 要调整图像的比例
          keep_ratio=False),  # 是否保持纵横比进行调整大小
      dict(  # Flip 的配置
          type='Flip',  # 翻转的流程
          flip_ratio=0.5),  # 实施翻转的概率
      dict(  # FormatShape 的配置
          type='FormatShape',  # 格式化形状的流程，将最终图像形状格式化为给定的 input_format
          input_format='NCHW'),  # 最终图像形状的格式
      dict(type='PackActionInputs')  # PackActionInputs 的配置
  ]
  val_pipeline = [  # 验证数据处理流程
      dict(  # SampleFrames 的配置
          type='SampleFrames',  # 采样帧的流程，从视频中采样帧
          clip_len=1,  # 每个采样输出剪辑的帧数
          frame_interval=1,  # 相邻采样帧的时间间隔
          num_clips=3,  # 要采样的剪辑数
          test_mode=True),  # 是否在采样时设置为测试模式
      dict(  # RawFrameDecode 的配置
          type='RawFrameDecode'),  # 加载和解码帧的流程，选择给定索引的原始帧
      dict(  # Resize 的配置
          type='Resize',  # 调整大小的流程
          scale=(-1, 256)),  # 要调整图像的比例
      dict(  # CenterCrop 的配置
          type='CenterCrop',  # 中心裁剪的流程，从图像中裁剪中心区域
          crop_size=224),  # 要裁剪的图像大小
      dict(  # Flip 的配置
          type='Flip',  # 翻转的流程
          flip_ratio=0),  # 实施翻转的概率
      dict(  # FormatShape 的配置
          type='FormatShape',  # 格式化形状的流程，将最终图像形状格式化为给定的 input_format
          input_format='NCHW'),  # 最终图像形状的格式
      dict(type='PackActionInputs')  # PackActionInputs 的配置
  ]
  test_pipeline = [  # 测试数据处理流程
      dict(  # SampleFrames 的配置
          type='SampleFrames',  # 采样帧的流程，从视频中采样帧
          clip_len=1,  # 每个采样输出剪辑的帧数
          frame_interval=1,  # 相邻采样帧的时间间隔
          num_clips=25,  # 要采样的剪辑数
          test_mode=True),  # 是否在采样时设置为测试模式
      dict(  # RawFrameDecode 的配置
          type='RawFrameDecode'),  # 加载和解码帧的流程，选择给定索引的原始帧
      dict(  # Resize 的配置
          type='Resize',  # 调整大小的流程
          scale=(-1, 256)),  # 要调整图像的比例
      dict(  # TenCrop 的配置
          type='TenCrop',  # 十次裁剪的流程，从图像中裁剪十个区域
          crop_size=224),  # 要裁剪的图像大小
      dict(  # Flip 的配置
          type='Flip',  # 翻转的流程
          flip_ratio=0),  # 实施翻转的概率
      dict(  # FormatShape 的配置
          type='FormatShape',  # 格式化形状的流程，将最终图像形状格式化为给定的 input_format
          input_format='NCHW'),  # 最终图像形状的格式
      dict(type='PackActionInputs')  # PackActionInputs 的配置
  ]

  train_dataloader = dict(  # 训练数据加载器的配置
      batch_size=32,  # 训练时每个单个 GPU 的批量大小
      num_workers=8,  # 训练时每个单个 GPU 的数据预取进程数
      persistent_workers=True,  # 如果为 `True`，则数据加载器在一个 epoch 结束后不会关闭工作进程，这可以加速训练速度
      sampler=dict(
          type='DefaultSampler',  # 支持分布式和非分布式训练的 DefaultSampler。参考 https://github.com/open-mmlab/mmengine/blob/main/mmengine/dataset/sampler.py
          shuffle=True),  # 每个 epoch 随机打乱训练数据
      dataset=dict(  # 训练数据集的配置
          type=dataset_type,
          ann_file=ann_file_train,  # 注释文件的路径
          data_prefix=dict(img=data_root),  # 帧路径的前缀
          pipeline=train_pipeline))
  val_dataloader = dict(  # 验证数据加载器的配置
      batch_size=1,  # 验证时每个单个 GPU 的批量大小
      num_workers=8,  # 验证时每个单个 GPU 的数据预取进程数
      persistent_workers=True,  # 如果为 `True`，则数据加载器在一个 epoch 结束后不会关闭工作进程
      sampler=dict(
          type='DefaultSampler',
          shuffle=False),  # 验证和测试时不进行随机打乱
      dataset=dict(  # 验证数据集的配置
          type=dataset_type,
          ann_file=ann_file_val,  # 注释文件的路径
          data_prefix=dict(img=data_root_val),  # 帧路径的前缀
          pipeline=val_pipeline,
          test_mode=True))
  test_dataloader = dict(  # 测试数据加载器的配置
      batch_size=32,  # 测试时每个单个 GPU 的批量大小
      num_workers=8,  # 测试时每个单个 GPU 的数据预取进程数
      persistent_workers=True,  # 如果为 `True`，则数据加载器在一个 epoch 结束后不会关闭工作进程
      sampler=dict(
          type='DefaultSampler',
          shuffle=False),  # 验证和测试时不进行随机打乱
      dataset=dict(  # 测试数据集的配置
          type=dataset_type,
          ann_file=ann_file_val,  # 注释文件的路径
          data_prefix=dict(img=data_root_val),  # 帧路径的前缀
          pipeline=test_pipeline,
          test_mode=True))

  # 评估设置
  val_evaluator = dict(type='AccMetric')  # 验证评估器的配置
  test_evaluator = val_evaluator  # 测试评估器的配置

  train_cfg = dict(  # 训练循环的配置
      type='EpochBasedTrainLoop',  # 训练循环的名称
      max_epochs=100,  # 总的训练周期数
      val_begin=1,  # 开始验证的训练周期
      val_interval=1)  # 验证间隔
  val_cfg = dict(  # 验证循环的配置
      type='ValLoop')  # 验证循环的名称
  test_cfg = dict( # 测试循环的配置
      type='TestLoop')  # 测试循环的名称

  # 学习策略
  param_scheduler = [  # 更新优化器参数的学习率测率，支持字典或列表
      dict(type='MultiStepLR',  # 达到一个里程碑时衰减学习率
          begin=0,  # 开始更新学习率的步骤
          end=100,  # 结束更新学习率的步骤
          by_epoch=True,  # 是否按 epoch 更新学习率
          milestones=[40, 80],  # 衰减学习率的步骤
          gamma=0.1)]  # 学习率衰减的乘法因子

  # 优化器
  optim_wrapper = dict(  # 优化器包装器的配置
      type='OptimWrapper',  # 优化器包装器的名称，切换到 AmpOptimWrapper 可以启用混合精度训练
      optimizer=dict(  # 优化器的配置。支持 PyTorch 中的各种优化器。参考 https://pytorch.org/docs/stable/optim.html#algorithms
          type='SGD',  # 优化器的名称
          lr=0.01,  # 学习率
          momentum=0.9,  # 动量因子
          weight_decay=0.0001),  # 权重衰减
      clip_grad=dict(max_norm=40, norm_type=2))  # 梯度裁剪的配置

  # 运行时设置
  default_scope = 'mmaction'  # 用于查找模块的默认注册表作用域。参考 https://mmengine.readthedocs.io/en/latest/tutorials/registry.html
  default_hooks = dict(  # 执行默认操作的钩子，如更新模型参数和保存权重。
      runtime_info=dict(type='RuntimeInfoHook'),  # 将运行时信息更新到消息中心的钩子
      timer=dict(type='IterTimerHook'),  # 用于记录迭代过程中花费的时间的日志记录器
      logger=dict(
          type='LoggerHook',  # 用于记录训练/验证/测试阶段的日志记录器
          interval=20,  # 打印日志的间隔
          ignore_last=False), # 忽略每个 epoch 中最后几个迭代的日志
      param_scheduler=dict(type='ParamSchedulerHook'),  # 更新优化器中某些超参数的钩子
      checkpoint=dict(
          type='CheckpointHook',  # 定期保存权重的钩子
          interval=3,  # 保存的周期
          save_best='auto',  # 用于评估最佳权重的指标
          max_keep_ckpts=3),  # 保留的最大权重文件数量
      sampler_seed=dict(type='DistSamplerSeedHook'),  # 用于分布式训练的数据加载采样器
      sync_buffers=dict(type='SyncBuffersHook'))  # 在每个 epoch 结束时同步模型缓冲区

  env_cfg = dict(  # 设置环境的字典
      cudnn_benchmark=False,  # 是否启用 cudnn benchmark
      mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0), # 设置多进程的参数
      dist_cfg=dict(backend='nccl')) # 设置分布式环境的参数，也可以设置端口号

  log_processor = dict(
      type='LogProcessor',  # 用于格式化日志信息的日志处理器
      window_size=20,  # 默认的平滑间隔
      by_epoch=True)  # 是否使用 epoch 类型格式化日志
  vis_backends = [  # 可视化后端的列表
      dict(type='LocalVisBackend')]  # 本地可视化后端
  visualizer = dict(  # 可视化器的配置
      type='ActionVisualizer',  # 可视化器的名称
      vis_backends=vis_backends)
  log_level = 'INFO'  # 日志记录的级别
  load_from = None  # 从给定路径加载模型权重作为预训练模型。这不会恢复训练。
  resume = False  # 是否从 `load_from` 中定义的权重恢复。如果 `load_from` 为 None，则会从 `work_dir` 中恢复最新的权重。
  ```

### 时空动作检测的配置系统

我们将模块化设计融入我们的配置系统中，这方便进行各种实验。

- FastRCNN 的示例

  为了帮助用户对完整的配置结构和时空动作检测系统中的模块有一个基本的了解，
  我们对 FastRCNN 的配置进行简要注释如下。有关每个模块中每个参数的更详细用法和替代方法，请参阅 API 文档。

  ```python
  # 模型设置
  model = dict(  # 模型的配置
      type='FastRCNN',  # 检测器的类名
      _scope_='mmdet',  # 当前配置的范围
      backbone=dict(  # 骨干网络的配置
          type='ResNet3dSlowOnly',  # 骨干网络的名称
          depth=50, # ResNet 模型的深度
          pretrained=None,   # 预训练模型的 URL/网站
          pretrained2d=False, # 如果预训练模型是 2D 的
          lateral=False,  # 如果骨干网络带有横向连接
          num_stages=4, # ResNet 模型的阶段数
          conv1_kernel=(1, 7, 7), # Conv1 的卷积核大小
          conv1_stride_t=1, # Conv1 的时间步长
          pool1_stride_t=1, # Pool1 的时间步长
          spatial_strides=(1, 2, 2, 1)),  # 每个 ResNet 阶段的空间步长
      roi_head=dict(  # roi_head 的配置
          type='AVARoIHead',  # roi_head 的名称
          bbox_roi_extractor=dict(  # bbox_roi_extractor 的配置
              type='SingleRoIExtractor3D',  # bbox_roi_extractor 的名称
              roi_layer_type='RoIAlign',  # RoI 操作的类型
              output_size=8,  # RoI 操作的输出特征大小
              with_temporal_pool=True), # 是否进行时间维度的池化
          bbox_head=dict( # bbox_head 的配置
              type='BBoxHeadAVA', # bbox_head 的名称
              in_channels=2048, # 输入特征的通道数
              num_classes=81, # 动作类别数 + 1
              multilabel=True,  # 数据集是否为多标签
              dropout_ratio=0.5),  # 使用的 dropout 比例
      data_preprocessor=dict(  # 数据预处理器的配置
          type='ActionDataPreprocessor',  # 数据预处理器的名称
          mean=[123.675, 116.28, 103.53],  # 不同通道的均值用于归一化
          std=[58.395, 57.12, 57.375],  # 不同通道的标准差用于归一化
          format_shape='NCHW'))  # 最终图像形状的格式
      train_cfg=dict(
          rcnn=dict(
              assigner=dict(
                  type='MaxIoUAssignerAVA',  # 分配器的名称
                  pos_iou_thr=0.9,  # 正样本的 IoU 阈值，> pos_iou_thr -> 正样本
                  neg_iou_thr=0.9,  # 负样本的 IoU 阈值，< neg_iou_thr -> 负样本
                  min_pos_iou=0.9),  # 正样本的最小可接受 IoU
              sampler=dict(
                  type='RandomSampler',  # 采样器的名称
                  num=32,  # 采样器的批处理大小
                  pos_fraction=1,  # 采样器的正样本比例
                  neg_pos_ub=-1,  # 负样本与正样本数量比率的上限
                  add_gt_as_proposals=True),  # 将 gt 边界框添加到 proposals 中
              pos_weight=1.0)),  # 正样本的损失权重
      test_cfg=dict(rcnn=None))  # 测试的配置

  # 数据集设置
  dataset_type = 'AVADataset'  # 训练、验证和测试的数据集类型
  data_root = 'data/ava/rawframes'  # 数据的根目录
  anno_root = 'data/ava/annotations'  # 注释的根目录

  ann_file_train = f'{anno_root}/ava_train_v2.1.csv'  # 训练注释文件的路径
  ann_file_val = f'{anno_root}/ava_val_v2.1.csv'  # 验证注释文件的路径

  exclude_file_train = f'{anno_root}/ava_train_excluded_timestamps_v2.1.csv'  # 训练排除注释文件的路径
  exclude_file_val = f'{anno_root}/ava_val_excluded_timestamps_v2.1.csv'  # 验证排除注释文件的路径

  label_file = f'{anno_root}/ava_action_list_v2.1_for_activitynet_2018.pbtxt'  # 标签文件的路径

  proposal_file_train = f'{anno_root}/ava_dense_proposals_train.FAIR.recall_93.9.pkl'  # 训练示例的人体检测 proposals 文件的路径
  proposal_file_val = f'{anno_root}/ava_dense_proposals_val.FAIR.recall_93.9.pkl'  # 验证示例的人体检测 proposals 文件的路径

  train_pipeline = [
      dict(
          type='AVASampleFrames',  # 从视频中采样帧的管道
          clip_len=4,  # 每个采样输出的帧数
          frame_interval=16),  # 相邻采样帧之间的时间间隔
      dict(
          type='RawFrameDecode'),  # 加载和解码帧的管道，使用给定的索引选择原始帧
      dict(
          type='RandomRescale',  # 随机缩放短边
          scale_range=(256, 320)),  # 随机缩放的短边尺寸范围
      dict(
          type='RandomCrop',  # 随机裁剪给定大小的补丁
          size=256),  # 裁剪补丁的大小
      dict(
          type='Flip',  # 翻转管道
          flip_ratio=0.5),  # 翻转的概率
      dict(
          type='FormatShape',  # 格式化形状的管道，将最终图像形状格式化为给定的输入格式
          input_format='NCTHW',  # 最终图像形状的格式
          collapse=True),  # 如果 N == 1，则减少维度 N
      dict(type='PackActionInputs')  # 打包输入数据
  ]

  val_pipeline = [
      dict(
          type='AVASampleFrames',  # 从视频中采样帧的管道
          clip_len=4,  # 每个采样输出的帧数
          frame_interval=16),  # 相邻采样帧之间的时间间隔
      dict(
          type='RawFrameDecode'),  # 加载和解码帧的管道，使用给定的索引选择原始帧
      dict(
          type='Resize',  # 调整大小的管道
          scale=(-1, 256)),  # 调整图像的尺度
      dict(
          type='FormatShape',  # 格式化形状的管道，将最终图像形状格式化为给定的输入格式
          input_format='NCTHW',  # 最终图像形状的格式
          collapse=True),  # 如果 N == 1，则减少维度 N
      dict(type='PackActionInputs')  # 打包输入数据
  ]

  train_dataloader = dict(
      batch_size=32,  # 每个单 GPU 训练的批处理大小
      num_workers=8,  # 每个单 GPU 训练时预取数据的 worker 数量
      persistent_workers=True,  # 如果为 `True`，则数据加载器在一个 epoch 结束后不会关闭 worker 进程，这可以加快训练速度
      sampler=dict(
          type='DefaultSampler',  # 默认采样器，支持分布式和非分布式训练。参考 https://github.com/open-mmlab/mmengine/blob/main/mmengine/dataset/sampler.py
          shuffle=True),  # 在每个 epoch 中随机打乱训练数据
      dataset=dict(
          type=dataset_type,
          ann_file=ann_file_train,  # 注释文件的路径
          exclude_file=exclude_file_train,  # 排除注释文件的路径
          label_file=label_file,  # 标签文件的路径
          data_prefix=dict(img=data_root),  # 帧路径的前缀
          proposal_file=proposal_file_train,  # 人体检测 proposals 的路径
          pipeline=train_pipeline)
  )
  val_dataloader = dict(
      batch_size=1,  # 每个单 GPU 评估的批处理大小
      num_workers=8,  # 每个单 GPU 评估时预取数据的 worker 数量
      persistent_workers=True,  # 如果为 `True`，则数据加载器在一个 epoch 结束后不会关闭 worker 进程
      sampler=dict(
          type='DefaultSampler',
          shuffle=False),  # 在验证和测试时不打乱数据
      dataset=dict(
          type=dataset_type,
          ann_file=ann_file_val,  # 注释文件的路径
          exclude_file=exclude_file_val,  # 排除注释文件的路径
          label_file=label_file,  # 标签文件的路径
          data_prefix=dict(img=data_root_val),  # 帧路径的前缀
          proposal_file=proposal_file_val,  # 人体检测 proposals 的路径
          pipeline=val_pipeline,
          test_mode=True)
  )
  test_dataloader = val_dataloader  # 测试数据加载器的配置

  # 评估设置
  val_evaluator = dict(
      type='AVAMetric',
      ann_file=ann_file_val,
      label_file=label_file,
      exclude_file=exclude_file_val)
  test_evaluator = val_evaluator  # 测试评估器的配置

  train_cfg = dict(
      type='EpochBasedTrainLoop',  # 训练循环的名称
      max_epochs=20,  # 总的训练 epoch 数量
      val_begin=1,  # 开始验证的 epoch
      val_interval=1)  # 验证的间隔
  val_cfg = dict(
      type='ValLoop')  # 验证循环的名称
  test_cfg = dict(
      type='TestLoop')  # 测试循环的名称

  # 学习策略
  param_scheduler = [
      dict(
          type='LinearLR',  # 线性减少每个参数组的学习率
          start_factor=0.1,  # 第一个 epoch 中学习率的乘法因子
          by_epoch=True,  # 是否按 epoch 更新学习率
          begin=0,  # 开始更新学习率的步骤
          end=5),  # 停止更新学习率的步骤
      dict(
          type='MultiStepLR',  # 当 epoch 数达到里程碑时，减少学习率
          begin=0,  # 开始更新学习率的步骤
          end=20,  # 停止更新学习率的步骤
          by_epoch=True,  # 是否按 epoch 更新学习率
          milestones=[10, 15],  # 学习率衰减的步骤
          gamma=0.1)  # 学习率衰减的乘法因子
  ]

  # 优化器
  optim_wrapper = dict(
      type='OptimWrapper',  # 优化器包装器的名称，切换到 AmpOptimWrapper 以启用混合精度训练
      optimizer=dict(
          type='SGD',  # 优化器的名称
          lr=0.2,  # 学习率
          momentum=0.9,  # 动量因子
          weight_decay=0.0001),  # 权重衰减
      clip_grad=dict(max_norm=40, norm_type=2))  # 梯度剪裁的配置

  # 运行时设置
  default_scope = 'mmaction'  # 默认注册表范围，用于查找模块。参考 https://mmengine.readthedocs.io/en/latest/tutorials/registry.html
  default_hooks = dict(
      runtime_info=dict(type='RuntimeInfoHook'),  # 将运行时信息更新到消息中心的钩子
      timer=dict(type='IterTimerHook'),  # 用于记录迭代过程中花费的时间的日志记录器
      logger=dict(
          type='LoggerHook',  # 用于记录训练/验证/测试阶段的日志的日志记录器
          interval=20,  # 打印日志的间隔
          ignore_last=False),  # 忽略每个 epoch 中最后几次迭代的日志
      param_scheduler=dict(type='ParamSchedulerHook'),  # 更新优化器中的某些超参数的钩子
      checkpoint=dict(
          type='CheckpointHook',  # 定期保存权重的钩子
          interval=3,  # 保存周期
          save_best='auto',  # 在评估过程中测量最佳权重的指标
          max_keep_ckpts=3),  # 保留的最大权重文件数量
      sampler_seed=dict(type='DistSamplerSeedHook'),  # 用于分布式训练的数据加载采样器
      sync_buffers=dict(type='SyncBuffersHook'))  # 在每个 epoch 结束时同步模型缓冲区的钩子
  env_cfg = dict(
      cudnn_benchmark=False,  # 是否启用 cudnn 的基准测试
      mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),  # 设置多进程的参数
      dist_cfg=dict(backend='nccl'))  # 设置分布式环境的参数，也可以设置端口

  log_processor = dict(
      type='LogProcessor',  # 用于格式化日志信息的日志处理器
      window_size=20,  # 默认平滑间隔
      by_epoch=True)  # 是否使用 epoch 类型格式化日志
  vis_backends = [
      dict(type='LocalVisBackend')]  # 可视化后端的列表
  visualizer = dict(
      type='ActionVisualizer',  # 可视化器的名称
      vis_backends=vis_backends)
  log_level = 'INFO'  # 日志级别
  load_from = ('https://download.openmmlab.com/mmaction/v1.0/recognition/slowonly/'
              'slowonly_imagenet-pretrained-r50_8xb16-4x16x1-steplr-150e_kinetics400-rgb/'
              'slowonly_imagenet-pretrained-r50_8xb16-4x16x1-steplr-150e_kinetics400-rgb_20220901-e7b65fad.pth')  # 从给定路径加载模型权重作为预训练模型。这不会恢复训练。
  resume = False  # 是否从 `load_from` 中定义的权重恢复训练。如果 `load_from` 为 None，则会从 `work_dir` 中恢复最新的权重。
  ```

### 动作定位的配置系统

我们将模块化设计引入了配置系统中，方便进行各种实验。

- BMN 的示例

  为了帮助用户对完整的配置结构和动作定位系统中的模块有一个基本的了解，我们对 BMN 的配置进行了简要注释，具体如下所示。有关每个模块中每个参数的更详细用法和替代方法，请参阅 [API 文档](https://mmaction2.readthedocs.io/en/latest/api.html)。

  ```python
  # 模型设置
  model = dict(
      type='BMN',  # 定位器的类名
      temporal_dim=100,  # 每个视频选取的总帧数
      boundary_ratio=0.5,  # 确定视频边界的比率
      num_samples=32,  # 每个 proposal 的采样数量
      num_samples_per_bin=3,  # 每个采样的 bin 的采样数量
      feat_dim=400,  # 特征的维度
      soft_nms_alpha=0.4,  # Soft NMS 的 alpha 值
      soft_nms_low_threshold=0.5,  # Soft NMS 的低阈值
      soft_nms_high_threshold=0.9,  # Soft NMS 的高阈值
      post_process_top_k=100)  # 后处理中的 top-k proposal 数量

  # 数据集设置
  dataset_type = 'ActivityNetDataset'  # 用于训练、验证和测试的数据集类型
  data_root = 'data/activitynet_feature_cuhk/csv_mean_100/'  # 用于训练的数据的根目录
  data_root_val = 'data/activitynet_feature_cuhk/csv_mean_100/'  # 用于验证和测试的数据的根目录
  ann_file_train = 'data/ActivityNet/anet_anno_train.json'  # 用于训练的注释文件的路径
  ann_file_val = 'data/ActivityNet/anet_anno_val.json'  # 用于验证的注释文件的路径
  ann_file_test = 'data/ActivityNet/anet_anno_test.json'  # 用于测试的注释文件的路径

  train_pipeline = [
      dict(type='LoadLocalizationFeature'),  # 加载定位特征的管道
      dict(type='GenerateLocalizationLabels'),  # 生成定位标签的管道
      dict(
          type='PackLocalizationInputs',  # 打包定位数据
          keys=('gt_bbox'),  # 输入的键
          meta_keys=('video_name'))]  # 输入的元键
  val_pipeline = [
      dict(type='LoadLocalizationFeature'),  # 加载定位特征的管道
      dict(type='GenerateLocalizationLabels'),  # 生成定位标签的管道
      dict(
          type='PackLocalizationInputs',  # 打包定位数据
          keys=('gt_bbox'),   # 输入的键
          meta_keys=('video_name', 'duration_second', 'duration_frame',
                     'annotations', 'feature_frame'))]  # 输入的元键
  test_pipeline = [
      dict(type='LoadLocalizationFeature'),  # 加载定位特征的管道
      dict(
          type='PackLocalizationInputs',  # 打包定位数据
          keys=('gt_bbox'),  # 输入的键
          meta_keys=('video_name', 'duration_second', 'duration_frame',
                     'annotations', 'feature_frame'))]  # 输入的元键
  train_dataloader = dict(
      batch_size=8,  # 每个单 GPU 训练的批处理大小
      num_workers=8,  # 每个单 GPU 训练时预取数据的 worker 数量
      persistent_workers=True,  # 如果为 `True`，则数据加载器在一个 epoch 结束后不会关闭 worker 进程，这可以加快训练速度
      sampler=dict(
          type='DefaultSampler',  # 默认采样器，支持分布式和非分布式训练。参考 https://github.com/open-mmlab/mmengine/blob/main/mmengine/dataset/sampler.py
          shuffle=True),  # 在每个 epoch 中随机打乱训练数据
      dataset=dict(
          type=dataset_type,
          ann_file=ann_file_train,  # 注释文件的路径
          data_prefix=dict(video=data_root),  # 视频路径的前缀
          pipeline=train_pipeline)
  )
  val_dataloader = dict(
      batch_size=1,  # 每个单 GPU 评估的批处理大小
      num_workers=8,  # 每个单 GPU 评估时预取数据的 worker 数量
      persistent_workers=True,  # 如果为 `True`，则数据加载器在一个 epoch 结束后不会关闭 worker 进程
      sampler=dict(
          type='DefaultSampler',
          shuffle=False),  # 在验证和测试时不打乱数据
      dataset=dict(
          type=dataset_type,
          ann_file=ann_file_val,  # 注释文件的路径
          data_prefix=dict(video=data_root_val),  # 视频路径的前缀
          pipeline=val_pipeline,
          test_mode=True)
  )
  test_dataloader = dict(
      batch_size=1,  # 每个单 GPU 测试的批处理大小
      num_workers=8,  # 每个单 GPU 测试时预取数据的 worker 数量
      persistent_workers=True,  # 如果为 `True`，则数据加载器在一个 epoch 结束后不会关闭 worker 进程
      sampler=dict(
          type='DefaultSampler',
          shuffle=False),  # 在验证和测试时不打乱数据
      dataset=dict(
          type=dataset_type,
          ann_file=ann_file_val,  # 注释文件的路径
          data_prefix=dict(video=data_root_val),  # 视频路径的前缀
          pipeline=test_pipeline,
          test_mode=True)
  )

  # 评估设置
  work_dir = './work_dirs/bmn_400x100_2x8_9e_activitynet_feature/'  # 保存当前实验的模型权重和日志的目录
  val_evaluator = dict(
      type='ANetMetric',
      metric_type='AR@AN',
      dump_config=dict(
          out=f'{work_dir}/results.json',  # 输出文件的路径
          output_format='json'))  # 输出文件的格式
  test_evaluator = val_evaluator  # 将 test_evaluator 设置为 val_evaluator

  max_epochs = 9  # 训练模型的总 epoch 数量
  train_cfg = dict(
      type='EpochBasedTrainLoop',  # 训练循环的名称
      max_epochs=max_epochs,  # 总的训练 epoch 数量
      val_begin=1,  # 开始验证的 epoch
      val_interval=1)  # 验证的间隔
  val_cfg = dict(
      type='ValLoop')  # 验证循环的名称
  test_cfg = dict(
      type='TestLoop')  # 测试循环的名称

  # 学习策略
  param_scheduler = [
      dict(
          type='MultiStepLR',  # 当 epoch 数达到里程碑时，减少学习率
          begin=0,  # 开始更新学习率的步骤
          end=max_epochs,  # 停止更新学习率的步骤
          by_epoch=True,  # 是否按 epoch 更新学习率
          milestones=[7, ],  # 学习率衰减的步骤
          gamma=0.1)  # 学习率衰减的乘法因子
  ]

  # 优化器
  optim_wrapper = dict(
      type='OptimWrapper',  # 优化器包装器的名称，切换到 AmpOptimWrapper 以启用混合精度训练
      optimizer=dict(
          type='Adam',  # 优化器的名称
          lr=0.001,  # 学习率
          weight_decay=0.0001),  # 权重衰减
      clip_grad=dict(max_norm=40, norm_type=2))  # 梯度剪裁的配置

  # 运行时设置
  default_scope = 'mmaction'  # 默认注册表范围，用于查找模块。参考 https://mmengine.readthedocs.io/en/latest/tutorials/registry.html
  default_hooks = dict(
      runtime_info=dict(type='RuntimeInfoHook'),  # 将运行时信息更新到消息中心的钩子
      timer=dict(type='IterTimerHook'),  # 用于记录迭代过程中花费的时间的日志记录器
      logger=dict(
          type='LoggerHook',  # 用于记录训练/验证/测试阶段的日志的日志记录器
          interval=20,  # 打印日志的间隔
          ignore_last=False),  # 忽略每个 epoch 中最后几次迭代的日志
      param_scheduler=dict(type='ParamSchedulerHook'),  # 更新优化器中的某些超参数的钩子
      checkpoint=dict(
          type='CheckpointHook',  # 定期保存权重的钩子
          interval=3,  # 保存周期
          save_best='auto',  # 在评估过程中测量最佳权重的指标
          max_keep_ckpts=3),  # 保留的最大权重文件数量
      sampler_seed=dict(type='DistSamplerSeedHook'),  # 用于分布式训练的数据加载采样器
      sync_buffers=dict(type='SyncBuffersHook'))  # 在每个 epoch 结束时同步模型缓冲区的钩子
  env_cfg = dict(
      cudnn_benchmark=False,  # 是否启用 cudnn 的基准测试
      mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),  # 设置多进程的参数
      dist_cfg=dict(backend='nccl'))  # 设置分布式环境的参数，也可以设置端口

  log_processor = dict(
      type='LogProcessor',  # 用于格式化日志信息的日志处理器
      window_size=20,  # 默认平滑间隔
      by_epoch=True)  # 是否使用 epoch 类型格式化日志
  vis_backends = [
      dict(type='LocalVisBackend')]  # 可视化后端的列表
  visualizer = dict(
      type='ActionVisualizer',  # 可视化器的名称
      vis_backends=vis_backends)
  log_level = 'INFO'  # 日志级别
  load_from = None  # 从给定路径加载模型权重作为预训练模型。这不会恢复训练。
  resume = False  # 是否从 `load_from` 中定义的权重恢复训练。如果 `load_from` 为 None，则会从 `work_dir` 中恢复最新的权重。
  ```
