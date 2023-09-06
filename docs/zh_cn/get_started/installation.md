# 安装

## 前置条件

在本节中，我们将演示如何准备 PyTorch 相关的依赖环境。

MMAction2 适用于 Linux、Windows 和 MacOS。它需要 Python 3.7+，CUDA 10.2+ 和 PyTorch 1.8+。

```{note}
如果您熟悉 PyTorch 并且已经安装了它，可以跳过这部分内容，直接转到[下一节](#installation)。否则，您可以按照以下步骤进行准备工作。
```

**第一步。** 从[官方网站](https://docs.conda.io/en/latest/miniconda.html)下载并安装 Miniconda。

**第二步。** 创建一个 conda 环境并激活它。

```shell
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
```

**第三步。** 安装 PyTorch，按照[官方说明](https://pytorch.org/get-started/locally/)进行操作，例如：

在 GPU 平台上：

```shell
conda install pytorch torchvision -c pytorch
```

```{warning}
此命令将自动安装最新版本的 PyTorch 和 cudatoolkit，请确保它们与您的环境匹配。
```

在 CPU 平台上：

```shell
conda install pytorch torchvision cpuonly -c pytorch
```

## 最佳实践

我们建议用户遵循我们的最佳实践来安装 MMAction2。然而，整个过程是高度可定制的。更多信息请参见[自定义安装](#customize-installation)部分。

**第一步。** 使用 [MIM](https://github.com/open-mmlab/mim) 安装 [MMEngine](https://github.com/open-mmlab/mmengine)、[MMCV](https://github.com/open-mmlab/mmcv)、[MMDetection](https://github.com/open-mmlab/mmdetection)（可选）和 [MMPose](https://github.com/open-mmlab/mmpose)（可选）。

```shell
pip install -U openmim
mim install mmengine
mim install mmcv
mim install mmdet
mim install mmpose
```

**第二步。** 安装 MMAction2。

根据您的需求，我们支持两种安装模式：

- [从源代码构建 MMAction2（推荐）](#build-mmaction2-from-source)：您想在 MMAction2 框架上开发自己的动作识别任务或新功能。例如，添加新的数据集或新的模型。因此，您可以使用我们提供的所有工具。
- [安装为 Python 包](#install-as-a-python-package)：您只想在项目中调用 MMAction2 的 API 或导入 MMAction2 的模块。

### 从源代码构建 MMAction2

在这种情况下，从源代码安装 mmaction2：

```shell
git clone https://github.com/open-mmlab/mmaction2.git
cd mmaction2
pip install -v -e .
# "-v" 表示输出更多安装相关的信息
# "-e" 表示以可编辑形式安装，这样可以在不重新安装的情况下，让本地修改直接生效。
```

可选地，如果您希望为 MMAction2 做出贡献或体验实验功能，请切换到 `dev-1.x` 分支：

```shell
git checkout dev-1.x
```

### 安装为 Python 包

只需使用 pip 安装即可。

```shell
pip install mmaction2
```

## 验证安装

为了验证 MMAction2 是否安装正确，我们提供了一些示例代码来运行推理演示。

**第一步。** 下载配置文件和权重文件。

```shell
mim download mmaction2 --config tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb --dest .
```

**第二步。** 验证推理演示。

选项（a）。如果您是从源代码安装的 mmaction2，可以运行以下命令：

```shell
# demo.mp4 和 label_map_k400.txt 都来自于 Kinetics-400
python demo/demo.py tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb.py \
    tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb_20220906-2692d16c.pth \
    demo/demo.mp4 tools/data/kinetics/label_map_k400.txt
```

您将在终端看到前5个标签及其对应的分数。

选项（b）。如果您将 mmaction2 安装为一个 Python 包，可以在 Python 解释器中运行以下代码，这将进行类似的验证：

```python
from operator import itemgetter
from mmaction.apis import init_recognizer, inference_recognizer

config_file = 'tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb.py'
checkpoint_file = 'tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb_20220906-2692d16c.pth'
video_file = 'demo/demo.mp4'
label_file = 'tools/data/kinetics/label_map_k400.txt'
model = init_recognizer(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'
pred_result = inference_recognizer(model, video_file)

pred_scores = pred_result.pred_score.tolist()
score_tuples = tuple(zip(range(len(pred_scores)), pred_scores))
score_sorted = sorted(score_tuples, key=itemgetter(1), reverse=True)
top5_label = score_sorted[:5]

labels = open(label_file).readlines()
labels = [x.strip() for x in labels]
results = [(labels[k[0]], k[1]) for k in top5_label]

print('The top-5 labels with corresponding scores are:')
for result in results:
    print(f'{result[0]}: ', result[1])
```

## 自定义安装

### CUDA 版本

在安装 PyTorch 时，您可能需要指定 CUDA 的版本。如果您不确定选择哪个版本，请遵循我们的建议：

- 对于 Ampere 架构的 NVIDIA GPU，例如 GeForce 30 series 以及 NVIDIA A100，CUDA 11 是必需的。
- 对于更早的 NVIDIA GPU，CUDA 11 是向前兼容的，但 CUDA 10.2 能够提供更好的兼容性，也更加轻量。

请确保 GPU 驱动程序满足最低版本要求。有关更多信息，请参见[此表格](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions__table-cuda-toolkit-driver-versions)。

```{note}
如果按照我们的最佳实践进行安装，仅安装 CUDA 运行时库就足够了，因为不会在本地编译任何 CUDA 代码。然而，如果您希望从源代码编译 MMCV 或开发其他 CUDA 运算符，您需要从 NVIDIA 的[网站](https://developer.nvidia.com/cuda-downloads)安装完整的 CUDA 工具包，并且其版本应与 PyTorch 的 CUDA 版本匹配，即 `conda install` 命令中指定的 cudatoolkit 的版本。
```

### 不使用 MIM 安装 MMCV

MMCV 包含 C++ 和 CUDA 扩展，因此它与 PyTorch 的关系比较复杂。MIM 可以自动解决这些依赖关系，使安装变得更加容易。但这不是必须的。

如果您希望使用 pip 而不是 MIM 安装 MMCV，请参考[MMCV 安装指南](https://mmcv.readthedocs.io/en/latest/get_started/installation.html)。这需要手动指定基于 PyTorch 版本和其 CUDA 版本的 find-url。

例如，以下命令安装了为 PyTorch 1.10.x 和 CUDA 11.3 构建的 mmcv。

```shell
pip install mmcv -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html
```

### 在 CPU 环境中安装

MMAction2 可以仅在 CPU 环境中安装。在 CPU 模式下，你可以完成训练、测试和模型推理等所有操作。

在 CPU 模式下，MMCV 的部分功能将不可用，通常是一些 GPU 编译的算子。不过不用担心， MMAction2 中几乎所有的模型都不会依赖这些算子。

### 通过 Docker 使用 MMAction2

我们提供了一个[Dockerfile](https://github.com/open-mmlab/mmaction2/blob/main/docker/Dockerfile)来构建镜像。确保您的[docker 版本](https://docs.docker.com/engine/install/) >=19.03。

```shell
# 构建一个基于 PyTorch 1.6.0、CUDA 10.1 和 CUDNN 7 的镜像。
# 如果您喜欢其他版本，请修改 Dockerfile。
docker build -f ./docker/Dockerfile --rm -t mmaction2 .
```

使用以下命令运行它：

```shell
# 例如构建PyTorch 1.6.0, CUDA 10.1, CUDNN 7的镜像
# 如果你喜欢其他版本,只要修改Dockerfile
docker run --gpus all --shm-size=8g -it -v {DATA_DIR}:/mmaction2/data mmaction2
```

## 故障排除

1. 当从旧版本 `0.x` 迁移到新版本 `1.x` 时，您可能会遇到依赖库版本不匹配的问题。下面是在按照上述安装过程执行后，通过 `pip list` 命令显示的每个依赖库的版本。请确保在终端中显示的每个依赖库版本都大于或等于（即 `>=`）下面每个依赖库的版本。

```shell
mmaction2                1.0.0
mmcv                     2.0.0
mmdet                    3.0.0
mmengine                 0.7.2
mmpose                   1.0.0
```
