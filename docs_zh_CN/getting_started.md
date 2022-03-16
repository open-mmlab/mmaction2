# 基础教程

本文档提供 MMAction2 相关用法的基本教程。对于安装说明，请参阅 [安装指南](install.md)。

<!-- TOC -->

- [基础教程](#基础教程)
  - [数据集](#数据集)
  - [使用预训练模型进行推理](#使用预训练模型进行推理)
    - [测试某个数据集](#测试某个数据集)
    - [使用高级 API 对视频和帧文件夹进行测试](#使用高级-api-对视频和帧文件夹进行测试)
  - [如何建立模型](#如何建立模型)
    - [使用基本组件建立模型](#使用基本组件建立模型)
    - [构建新模型](#构建新模型)
  - [如何训练模型](#如何训练模型)
    - [推理流水线](#推理流水线)
    - [训练配置](#训练配置)
    - [使用单个 GPU 进行训练](#使用单个-gpu-进行训练)
    - [使用多个 GPU 进行训练](#使用多个-gpu-进行训练)
    - [使用多台机器进行训练](#使用多台机器进行训练)
    - [使用单台机器启动多个任务](#使用单台机器启动多个任务)
  - [详细教程](#详细教程)

<!-- TOC -->

## 数据集

MMAction2 建议用户将数据集根目录链接到 `$MMACTION2/data` 下。
如果用户的文件夹结构与默认结构不同，则需要在配置文件中进行对应路径的修改。

```
mmaction2
├── mmaction
├── tools
├── configs
├── data
│   ├── kinetics400
│   │   ├── rawframes_train
│   │   ├── rawframes_val
│   │   ├── kinetics_train_list.txt
│   │   ├── kinetics_val_list.txt
│   ├── ucf101
│   │   ├── rawframes_train
│   │   ├── rawframes_val
│   │   ├── ucf101_train_list.txt
│   │   ├── ucf101_val_list.txt
│   ├── ...
```

请参阅 [数据集准备](data_preparation.md) 获取数据集准备的相关信息。

对于用户自定义数据集的准备，请参阅 [教程 3：如何增加新数据集](tutorials/3_new_dataset.md)

## 使用预训练模型进行推理

MMAction2 提供了一些脚本用于测试数据集（如 Kinetics-400，Something-Something V1&V2，(Multi-)Moments in Time，等），
并提供了一些高级 API，以便更好地兼容其他项目。

MMAction2 支持仅使用 CPU 进行测试。然而，这样做的速度**非常慢**，用户应仅使用其作为无 GPU 机器上的 debug 手段。
如需使用 CPU 进行测试，用户需要首先使用命令 `export CUDA_VISIBLE_DEVICES=-1` 禁用机器上的 GPU （如有），然后使用命令 `python tools/test.py {OTHER_ARGS}` 直接调用测试脚本。

### 测试某个数据集

- [x] 支持单 GPU
- [x] 支持单节点，多 GPU
- [x] 支持多节点

用户可使用以下命令进行数据集测试

```shell
# 单 GPU 测试
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}] \
    [--gpu-collect] [--tmpdir ${TMPDIR}] [--options ${OPTIONS}] [--average-clips ${AVG_TYPE}] \
    [--launcher ${JOB_LAUNCHER}] [--local_rank ${LOCAL_RANK}] [--onnx] [--tensorrt]

# 多 GPU 测试
./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}] \
    [--gpu-collect] [--tmpdir ${TMPDIR}] [--options ${OPTIONS}] [--average-clips ${AVG_TYPE}] \
    [--launcher ${JOB_LAUNCHER}] [--local_rank ${LOCAL_RANK}]
```

可选参数:

- `RESULT_FILE`：输出结果文件名。如果没有被指定，则不会保存测试结果。
- `EVAL_METRICS`：测试指标。其可选值与对应数据集相关，如 `top_k_accuracy`，`mean_class_accuracy` 适用于所有动作识别数据集，`mmit_mean_average_precision` 适用于 Multi-Moments in Time 数据集，`mean_average_precision` 适用于 Multi-Moments in Time 和单类 HVU 数据集，`AR@AN` 适用于 ActivityNet 数据集等。
- `--gpu-collect`：如果被指定，动作识别结果将会通过 GPU 通信进行收集。否则，它将被存储到不同 GPU 上的 `TMPDIR` 文件夹中，并在 rank 0 的进程中被收集。
- `TMPDIR`：用于存储不同进程收集的结果文件的临时文件夹。该变量仅当 `--gpu-collect` 没有被指定时有效。
- `OPTIONS`：用于验证过程的自定义选项。其可选值与对应数据集的 `evaluate` 函数变量有关。
- `AVG_TYPE`：用于平均测试片段结果的选项。如果被设置为 `prob`，则会在平均测试片段结果之前施加 softmax 函数。否则，会直接进行平均。
- `JOB_LAUNCHER`：分布式任务初始化启动器选项。可选值有 `none`，`pytorch`，`slurm`，`mpi`。特别地，如果被设置为 `none`, 则会以非分布式模式进行测试。
- `LOCAL_RANK`：本地 rank 的 ID。如果没有被指定，则会被设置为 0。
- `--onnx`: 如果指定，将通过 onnx 模型推理获取预测结果，输入参数 `CHECKPOINT_FILE` 应为 onnx 模型文件。Onnx 模型文件由 `/tools/deployment/pytorch2onnx.py` 脚本导出。目前，不支持多 GPU 测试以及动态张量形状（Dynamic shape）。请注意，数据集输出与模型输入张量的形状应保持一致。同时，不建议使用测试时数据增强，如 `ThreeCrop`，`TenCrop`，`twice_sample` 等。
- `--tensorrt`: 如果指定，将通过 TensorRT 模型推理获取预测结果，输入参数 `CHECKPOINT_FILE` 应为 TensorRT 模型文件。TensorRT 模型文件由导出的 onnx 模型以及 TensorRT 官方模型转换工具生成。目前，不支持多 GPU 测试以及动态张量形状（Dynamic shape）。请注意，数据集输出与模型输入张量的形状应保持一致。同时，不建议使用测试时数据增强，如 `ThreeCrop`，`TenCrop`，`twice_sample` 等。

例子：

假定用户将下载的模型权重文件放置在 `checkpoints/` 目录下。

1. 在 Kinetics-400 数据集下测试 TSN （不存储测试结果为文件），并验证 `top-k accuracy` 和 `mean class accuracy` 指标

    ```shell
    python tools/test.py configs/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb.py \
        checkpoints/SOME_CHECKPOINT.pth \
        --eval top_k_accuracy mean_class_accuracy
    ```

2. 使用 8 块 GPU 在 Something-Something V1 下测试 TSN，并验证 `top-k accuracy` 指标

    ```shell
    ./tools/dist_test.sh configs/recognition/tsn/tsn_r50_1x1x8_50e_sthv1_rgb.py \
        checkpoints/SOME_CHECKPOINT.pth \
        8 --out results.pkl --eval top_k_accuracy
    ```

3. 在 slurm 分布式环境中测试 TSN 在 Kinetics-400 数据集下的 `top-k accuracy` 指标

    ```shell
    python tools/test.py configs/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb.py \
        checkpoints/SOME_CHECKPOINT.pth \
        --launcher slurm --eval top_k_accuracy
    ```

4. 在 Something-Something V1 下测试 onnx 格式的 TSN 模型，并验证 `top-k accuracy` 指标

    ```shell
    python tools/test.py configs/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb.py \
        checkpoints/SOME_CHECKPOINT.onnx \
        --eval top_k_accuracy --onnx
    ```

### 使用高级 API 对视频和帧文件夹进行测试

这里举例说明如何构建模型并测试给定视频

```python
import torch

from mmaction.apis import init_recognizer, inference_recognizer

config_file = 'configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py'
# 从模型库中下载检测点，并把它放到 `checkpoints/` 文件夹下
checkpoint_file = 'checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth'

# 指定设备
device = 'cuda:0' # or 'cpu'
device = torch.device(device)

 # 根据配置文件和检查点来建立模型
model = init_recognizer(config_file, checkpoint_file, device=device)

# 测试单个视频并显示其结果
video = 'demo/demo.mp4'
labels = 'tools/data/kinetics/label_map_k400.txt'
results = inference_recognizer(model, video, labels)

# 显示结果
labels = open('tools/data/kinetics/label_map_k400.txt').readlines()
labels = [x.strip() for x in labels]
results = [(labels[k[0]], k[1]) for k in results]

print(f'The top-5 labels with corresponding scores are:')
for result in results:
    print(f'{result[0]}: ', result[1])
```

这里举例说明如何构建模型并测试给定帧文件夹

```python
import torch

from mmaction.apis import init_recognizer, inference_recognizer

config_file = 'configs/recognition/tsn/tsn_r50_inference_1x1x3_100e_kinetics400_rgb.py'
# 从模型库中下载检测点，并把它放到 `checkpoints/` 文件夹下
checkpoint_file = 'checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth'

# 指定设备
device = 'cuda:0' # or 'cpu'
device = torch.device(device)

 # 根据配置文件和检查点来建立模型
model = init_recognizer(config_file, checkpoint_file, device=device, use_frames=True)

# 测试单个视频的帧文件夹并显示其结果
video = 'SOME_DIR_PATH/'
labels = 'tools/data/kinetics/label_map_k400.txt'
results = inference_recognizer(model, video, labels, use_frames=True)

# 显示结果
labels = open('tools/data/kinetics/label_map_k400.txt').readlines()
labels = [x.strip() for x in labels]
results = [(labels[k[0]], k[1]) for k in results]

print(f'The top-5 labels with corresponding scores are:')
for result in results:
    print(f'{result[0]}: ', result[1])
```

这里举例说明如何构建模型并通过 url 测试给定视频

```python
import torch

from mmaction.apis import init_recognizer, inference_recognizer

config_file = 'configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py'
# 从模型库中下载检测点，并把它放到 `checkpoints/` 文件夹下
checkpoint_file = 'checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth'

# 指定设备
device = 'cuda:0' # or 'cpu'
device = torch.device(device)

 # 根据配置文件和检查点来建立模型
model = init_recognizer(config_file, checkpoint_file, device=device)

# 测试单个视频的 url 并显示其结果
video = 'https://www.learningcontainer.com/wp-content/uploads/2020/05/sample-mp4-file.mp4'
labels = 'tools/data/kinetics/label_map_k400.txt'
results = inference_recognizer(model, video, labels)

# 显示结果
labels = open('tools/data/kinetics/label_map_k400.txt').readlines()
labels = [x.strip() for x in labels]
results = [(labels[k[0]], k[1]) for k in results]

print(f'The top-5 labels with corresponding scores are:')
for result in results:
    print(f'{result[0]}: ', result[1])
```

**注意**：MMAction2 在默认提供的推理配置文件（inference configs）中定义 `data_prefix` 变量，并将其设置为 None 作为默认值。
如果 `data_prefix` 不为 None，则要获取的视频文件（或帧文件夹）的路径将为 `data_prefix/video`。
在这里，`video` 是上述脚本中的同名变量。可以在 `rawframe_dataset.py` 文件和 `video_dataset.py` 文件中找到此详细信息。例如，

- 当视频（帧文件夹）路径为 `SOME_DIR_PATH/VIDEO.mp4`（`SOME_DIR_PATH/VIDEO_NAME/img_xxxxx.jpg`），并且配置文件中的 `data_prefix` 为 None，则 `video` 变量应为 `SOME_DIR_PATH/VIDEO.mp4`（`SOME_DIR_PATH/VIDEO_NAME`）。
- 当视频（帧文件夹）路径为 `SOME_DIR_PATH/VIDEO.mp4`（`SOME_DIR_PATH/VIDEO_NAME/img_xxxxx.jpg`），并且配置文件中的 `data_prefix` 为 `SOME_DIR_PATH`，则 `video` 变量应为 `VIDEO.mp4`（`VIDEO_NAME`）。
- 当帧文件夹路径为 `VIDEO_NAME/img_xxxxx.jpg`，并且配置文件中的 `data_prefix` 为 None，则 `video` 变量应为 `VIDEO_NAME`。
- 当传递参数为视频 url 而非本地路径，则需使用 OpenCV 作为视频解码后端。

在 [demo/demo.ipynb](/demo/demo.ipynb) 中有提供相应的 notebook 演示文件。

## 如何建立模型

### 使用基本组件建立模型

MMAction2 将模型组件分为 4 种基础模型：

- 识别器（recognizer）：整个识别器模型管道，通常包含一个主干网络（backbone）和分类头（cls_head）。
- 主干网络（backbone）：通常为一个用于提取特征的 FCN 网络，例如 ResNet，BNInception。
- 分类头（cls_head）：用于分类任务的组件，通常包括一个带有池化层的 FC 层。
- 时序检测器（localizer）：用于时序检测的模型，目前有的检测器包含 BSN，BMN，SSN。

用户可参照给出的配置文件里的基础模型搭建流水线（如 `Recognizer2D`）

如果想创建一些新的组件，如 [TSM: Temporal Shift Module for Efficient Video Understanding](https://arxiv.org/abs/1811.08383) 中的 temporal shift backbone 结构，则需：

1. 创建 `mmaction/models/backbones/resnet_tsm.py` 文件

    ```python
    from ..builder import BACKBONES
    from .resnet import ResNet

    @BACKBONES.register_module()
    class ResNetTSM(ResNet):

      def __init__(self,
                   depth,
                   num_segments=8,
                   is_shift=True,
                   shift_div=8,
                   shift_place='blockres',
                   temporal_pool=False,
                   **kwargs):
          pass

      def forward(self, x):
          # implementation is ignored
          pass
    ```

2. 从 `mmaction/models/backbones/__init__.py` 中导入模型

    ```python
    from .resnet_tsm import ResNetTSM
    ```

3. 修改模型文件

    ```python
    backbone=dict(
      type='ResNet',
      pretrained='torchvision://resnet50',
      depth=50,
      norm_eval=False)
    ```

   修改为

    ```python
    backbone=dict(
        type='ResNetTSM',
        pretrained='torchvision://resnet50',
        depth=50,
        norm_eval=False,
        shift_div=8)
    ```

### 构建新模型

要编写一个新的动作识别器流水线，用户需要继承 `BaseRecognizer`，其定义了如下抽象方法

- `forward_train()`: 训练模式下的前向方法
- `forward_test()`: 测试模式下的前向方法

具体可参照 [Recognizer2D](/mmaction/models/recognizers/recognizer2d.py) 和 [Recognizer3D](/mmaction/models/recognizers/recognizer3d.py)

## 如何训练模型

### 推理流水线

MMAction2 使用 `MMDistributedDataParallel` 进行分布式训练，使用 `MMDataParallel` 进行非分布式训练。

对于单机多卡与多台机器的情况，MMAction2 使用分布式训练。假设服务器有 8 块 GPU，则会启动 8 个进程，并且每台 GPU 对应一个进程。

每个进程拥有一个独立的模型，以及对应的数据加载器和优化器。
模型参数同步只发生于最开始。之后，每经过一次前向与后向计算，所有 GPU 中梯度都执行一次 allreduce 操作，而后优化器将更新模型参数。
由于梯度执行了 allreduce 操作，因此不同 GPU 中模型参数将保持一致。

### 训练配置

所有的输出（日志文件和模型权重文件）会被将保存到工作目录下。工作目录通过配置文件中的参数 `work_dir` 指定。

默认情况下，MMAction2 在每个周期后会在验证集上评估模型，可以通过在训练配置中修改 `interval` 参数来更改评估间隔

```python
evaluation = dict(interval=5)  # 每 5 个周期进行一次模型评估
```

根据 [Linear Scaling Rule](https://arxiv.org/abs/1706.02677)，当 GPU 数量或每个 GPU 上的视频批大小改变时，用户可根据批大小按比例地调整学习率，如，当 4 GPUs x 2 video/gpu 时，lr=0.01；当 16 GPUs x 4 video/gpu 时，lr=0.08。

MMAction2 支持仅使用 CPU 进行训练。然而，这样做的速度**非常慢**，用户应仅使用其作为无 GPU 机器上的 debug 手段。
如需使用 CPU 进行训练，用户需要首先使用命令 `export CUDA_VISIBLE_DEVICES=-1` 禁用机器上的 GPU （如有），然后使用命令 `python tools/train.py {OTHER_ARGS}` 直接调用训练脚本。

### 使用单个 GPU 进行训练

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

如果用户想在命令中指定工作目录，则需要增加参数 `--work-dir ${YOUR_WORK_DIR}`

### 使用多个 GPU 进行训练

```shell
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

可选参数为：

- `--validate` (**强烈建议**)：在训练期间每 k 个周期进行一次验证（默认值为 5，可通过修改每个配置文件中的 `evaluation` 字典变量的 `interval` 值进行改变）。
- `--test-last`：在训练结束后使用最后一个检查点的参数进行测试，将测试结果存储在 `${WORK_DIR}/last_pred.pkl` 中。
- `--test-best`：在训练结束后使用效果最好的检查点的参数进行测试，将测试结果存储在 `${WORK_DIR}/best_pred.pkl` 中。
- `--work-dir ${WORK_DIR}`：覆盖配置文件中指定的工作目录。
- `--resume-from ${CHECKPOINT_FILE}`：从以前的模型权重文件恢复训练。
- `--gpus ${GPU_NUM}`：使用的 GPU 数量，仅适用于非分布式训练。
- `--gpu-ids ${GPU_IDS}`：使用的 GPU ID，仅适用于非分布式训练。
- `--seed ${SEED}`：设置 python，numpy 和 pytorch 里的种子 ID，已用于生成随机数。
- `--deterministic`：如果被指定，程序将设置 CUDNN 后端的确定化选项。
- `JOB_LAUNCHER`：分布式任务初始化启动器选项。可选值有 `none`，`pytorch`，`slurm`，`mpi`。特别地，如果被设置为 `none`, 则会以非分布式模式进行测试。
- `LOCAL_RANK`：本地 rank 的 ID。如果没有被指定，则会被设置为 0。

`resume-from` 和 `load-from` 的不同点：
`resume-from` 加载模型参数和优化器状态，并且保留检查点所在的周期数，常被用于恢复意外被中断的训练。
`load-from` 只加载模型参数，但周期数从 0 开始计数，常被用于微调模型。

这里提供一个使用 8 块 GPU 加载 TSN 模型权重文件的例子。

```shell
./tools/dist_train.sh configs/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb.py 8 --resume-from work_dirs/tsn_r50_1x1x3_100e_kinetics400_rgb/latest.pth
```

### 使用多台机器进行训练

如果用户在 [slurm](https://slurm.schedmd.com/) 集群上运行 MMAction2，可使用 `slurm_train.sh` 脚本。（该脚本也支持单台机器上进行训练）

```shell
[GPUS=${GPUS}] ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} [--work-dir ${WORK_DIR}]
```

这里给出一个在 slurm 集群上的 dev 分区使用 16 块 GPU 训练 TSN 的例子。（使用 `GPUS_PER_NODE=8` 参数来指定一个有 8 块 GPUS 的 slurm 集群节点）

```shell
GPUS=16 ./tools/slurm_train.sh dev tsn_r50_k400 configs/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb.py --work-dir work_dirs/tsn_r50_1x1x3_100e_kinetics400_rgb
```

用户可以查看 [slurm_train.sh](/tools/slurm_train.sh) 文件来检查完整的参数和环境变量。

如果您想使用由 ethernet 连接起来的多台机器， 您可以使用以下命令:

在第一台机器上:

```shell
NNODES=2 NODE_RANK=0 PORT=$MASTER_PORT MASTER_ADDR=$MASTER_ADDR sh tools/dist_train.sh $CONFIG $GPUS
```

在第二台机器上:

```shell
NNODES=2 NODE_RANK=1 PORT=$MASTER_PORT MASTER_ADDR=$MASTER_ADDR sh tools/dist_train.sh $CONFIG $GPUS
```

但是，如果您不使用高速网路连接这几台机器的话，训练将会非常慢。

### 使用单台机器启动多个任务

如果用使用单台机器启动多个任务，如在有 8 块 GPU 的单台机器上启动 2 个需要 4 块 GPU 的训练任务，则需要为每个任务指定不同端口，以避免通信冲突。

如果用户使用 `dist_train.sh` 脚本启动训练任务，则可以通过以下命令指定端口

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_train.sh ${CONFIG_FILE} 4
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 ./tools/dist_train.sh ${CONFIG_FILE} 4
```

如果用户在 slurm 集群下启动多个训练任务，则需要修改配置文件（通常是配置文件的倒数第 6 行）中的 `dist_params` 变量，以设置不同的通信端口。

在 `config1.py` 中，

```python
dist_params = dict(backend='nccl', port=29500)
```

在 `config2.py` 中，

```python
dist_params = dict(backend='nccl', port=29501)
```

之后便可启动两个任务，分别对应 `config1.py` 和 `config2.py`。

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=4 ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config1.py [--work-dir ${WORK_DIR}]
CUDA_VISIBLE_DEVICES=4,5,6,7 GPUS=4 ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config2.py [--work-dir ${WORK_DIR}]
```

## 详细教程

目前, MMAction2 提供以下几种更详细的教程：

- [如何编写配置文件](tutorials/1_config.md)
- [如何微调模型](tutorials/2_finetune.md)
- [如何增加新数据集](tutorials/3_new_dataset.md)
- [如何设计数据处理流程](tutorials/4_data_pipeline.md)
- [如何增加新模块](tutorials/5_new_modules.md)
- [如何导出模型为 onnx 格式](tutorials/6_export_model.md)
- [如何自定义模型运行参数](tutorials/7_customize_runtime.md)
