# 教程 6：如何导出模型为 onnx 格式

开放式神经网络交换格式（Open Neural Network Exchange，即 [ONNX](https://onnx.ai/)）是一个开放的生态系统，使 AI 开发人员能够随着项目的发展选择正确的工具。

<!-- TOC -->

- [教程 6：如何导出模型为 onnx 格式](#教程-6如何导出模型为-onnx-格式)
  - [支持的模型](#支持的模型)
  - [如何使用](#如何使用)
    - [准备工作](#准备工作)
    - [行为识别器](#行为识别器)
    - [时序动作检测器](#时序动作检测器)

<!-- TOC -->

## 支持的模型

到目前为止，MMAction2 支持将训练的 pytorch 模型中进行 onnx 导出。支持的模型有：

- I3D
- TSN
- TIN
- TSM
- R(2+1)D
- SLOWFAST
- SLOWONLY
- BMN
- BSN(tem, pem)

## 如何使用

对于简单的模型导出，用户可以使用这里的 [脚本](/tools/deployment/pytorch2onnx.py)。
注意，需要安装 `onnx` 和 `onnxruntime` 包以进行导出后的验证。

### 准备工作

首先，安装 onnx

```shell
pip install onnx onnxruntime
```

MMAction2 提供了一个 python 脚本，用于将 MMAction2 训练的 pytorch 模型导出到 ONNX。

```shell
python tools/deployment/pytorch2onnx.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--shape ${SHAPE}] \
    [--verify] [--show] [--output-file ${OUTPUT_FILE}]  [--is-localizer] [--opset-version ${VERSION}]
```

可选参数：

- `--shape`: 模型输入张量的形状。对于 2D 模型（如 TSN），输入形状应当为 `$batch $clip $channel $height $width` (例如，`1 1 3 224 224`)；对于 3D 模型（如 I3D），输入形状应当为 `$batch $clip $channel $time $height $width` (如，`1 1 3 32 224 224`)；对于时序检测器如 BSN，每个模块的数据都不相同，请查看对应的 `forward` 函数。如果没有被指定，它将被置为 `1 1 3 224 224`。
- `--verify`: 决定是否对导出模型进行验证，验证项包括是否可运行，数值是否正确等。如果没有被指定，它将被置为 `False`。
- `--show`: 决定是否打印导出模型的结构。如果没有被指定，它将被置为 `False`。
- `--output-file`: 导出的 onnx 模型名。如果没有被指定，它将被置为 `tmp.onnx`。
- `--is-localizer`：决定导出的模型是否为时序检测器。如果没有被指定，它将被置为 `False`。
- `--opset-version`：决定 onnx 的执行版本，MMAction2 推荐用户使用高版本（例如 11 版本）的 onnx 以确保稳定性。如果没有被指定，它将被置为 `11`。
- `--softmax`: 是否在行为识别器末尾添加 Softmax。如果没有指定，将被置为 `False`。目前仅支持行为识别器，不支持时序动作检测器。

### 行为识别器

对于行为识别器，可运行：

```shell
python tools/deployment/pytorch2onnx.py $CONFIG_PATH $CHECKPOINT_PATH --shape $SHAPE --verify
```

### 时序动作检测器

对于时序动作检测器，可运行：

```shell
python tools/deployment/pytorch2onnx.py $CONFIG_PATH $CHECKPOINT_PATH --is-localizer --shape $SHAPE --verify
```

如果发现提供的模型权重文件没有被成功导出，或者存在精度损失，可以在本 repo 下提出问题（issue）。
