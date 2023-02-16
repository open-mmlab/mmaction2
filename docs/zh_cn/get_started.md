# 前置条件

在本节中，我们将演示如何准备 PyTorch 相关的依赖环境。

MMAction2 适用于 Linux、Windows 和 MacOS。它需要 Python 3.7+，CUDA 9.2+ 和 PyTorch 1.6+。

```
如果你对配置 PyTorch 环境已经很熟悉，并且已经完成了配置，可以直接进入[下一节](#安装)。
否则的话，请依照以下步骤完成配置。
```

**第一步** 从[官网](https://docs.conda.io/en/latest/miniconda.html)下载并安装 Miniconda。

**第二步** 创建一个 conda 虚拟环境并激活它。

```shell
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
```

**第三步** 根据[官方指南](https://pytorch.org/get-started/locally/)安装 PyTorch。例如：

在GPU平台：

```shell
conda install pytorch torchvision -c pytorch
```

```
以上命令将自动安装最新版本的 PyTorch 和 cudatoolkit,请检查它们是否和你的环境匹配。
```

在CPU平台：

```shell
conda install pytorch torchvision cpuonly -c pytorch
```

# 安装

我们推荐用户按照我们的最佳实践安装 MMAction2。但除此之外，如果你想根据你的习惯完成安装，流程见[自定义安装](#自定义安装)章节获取更多信息。

## 最佳实践

**第一步** 使用 MIM 安装 MMEngine 和 MMCV。

```shell
pip install -U openmim
mim install mmengine 'mmcv>=2.0.0rc1'
```

请注意，MMAction2 中的一些推理示例脚本需要使用 [MMDetection](https://github.com/open-mmlab/mmdetection) (mmdet) 检测人体，[MMPose](https://github.com/open-mmlab/mmpose) 进行姿态估计。如果你想要运行这些示例脚本，可以通过以下命令安装 mmdet 和 mmpose:

```shell
mim install "mmdet>=3.0.0rc5"
mim install "mmpose>=1.0.0rc0"
```

**第二步** 安装 MMAction2。

根据你的需要，我们支持两种安装模式：

- [从源码安装（推荐）](#从源码安装)：希望开发自己的动作识别任务或者在 MMAction2 上开发新功能，例如，添加新的数据集或者新的模型。因此，你可以使用我们提供的所有工具。
- [作为 Python 包安装](#作为-Python-包安装)：只想希望调用 MMAction2 的 API 接口，或者在你的项目中导入 MMAction2 中的模块。

### 从源码安装

这种情况下，从源码按如下方式安装 MMAction2：

```shell
git clone https://github.com/open-mmlab/mmaction2.git
cd mmaction2
git checkout 1.x
pip install -v -e .
# "-v" 表示输出更多安装相关的信息
# "-e" 表示以可编辑形式安装，这样可以在不重新安装的情况下，让本地修改直接生效
```

另外，如果你想为 MMAction2 贡献代码，或者体验试验中的功能，请签出到 `dev-1.x` 分支。

```shell
git checkout dev-1.x
```

### 作为 Python 包安装

直接使用 pip 安装即可。

```shell
pip install "mmaction2>=1.0rc0"
```

## 验证安装

为了验证 MMAction2 的安装是否正确，我们提供了一些示例代码来执行模型推理。

**第一步**  我们需要下载配置文件和模型权重文件。

```shell
mim download mmaction2 --config tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb --dest .
```

**第二步**  验证示例的推理流程。

如果你从源码安装 MMAction2，那么直接运行以下命令进行验证：

```shell
# demo.mp4 和 label_map_k400.txt 都来自于 Kinetics-400
python demo/demo.py tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb.py \
    tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb_20220906-2692d16c.pth \
    demo/demo.mp4 tools/data/kinetics/label_map_k400.txt
```

终端上将输出获得最高分数的标签以及相应的分数。

如果你是作为 Python 包安装，那么可以打开你的 Python 解释器，并粘贴如下代码：

```python
from mmaction.apis import init_recognizer, inference_recognizer

config_file = 'tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb.py'
checkpoint_file = 'tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb_20220906-2692d16c.pth'
video_file = 'demo/demo.mp4'
label_file = 'tools/data/kinetics/label_map_k400.txt'
model = init_recognizer(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'
result = inference_recognizer(model, video_file)
pred_scores = result.pred_scores.item.tolist()
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

安装 PyTorch 时，你可能需要安装特定的 CUDA 的版本。如果你不清楚应该选择哪个版本，请遵循我们的建议：

- 对于 Ampere 架构的 NVIDIA GPU，例如 GeForce 30 series 以及 NVIDIA A100，CUDA 11 是必需的。
- 对于更早的 NVIDIA GPU，CUDA 11 是向前兼容的，但 CUDA 10.2 能够提供更好的兼容性，也更加轻量。

请确保你的 GPU 驱动满足要求的最低版本，详见[此表格](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions__table-cuda-toolkit-driver-versions)。

```
如果按照我们的最佳实践进行安装，CUDA 运行时库就足够了，因为我们提供相关 CUDA 代码的预编译，你不需要进行本地编译。
但如果你希望从源码进行 MMCV 的编译，或是进行其他 CUDA 算子的开发，那么就必须安装完整的 CUDA 工具链，参见
[NVIDIA 官网](https://developer.nvidia.com/cuda-downloads)，另外还需要确保该 CUDA 工具链的版本与 PyTorch 安装时
的配置相匹配（如用 `conda install` 安装 PyTorch 时指定的 cudatoolkit 版本）。
```

### 不使用 MIM 安装 MMCV

MMCV 包含 C++ 和 CUDA 扩展，因此其对 PyTorch 的依赖比较复杂。 MIM 会自动解析此类依赖关系，选择合适的 MMCV 预编译包，使安装更简单，但它并不是必需的。

要使用 pip 而不是 MIM 安装 MMCV，请遵循 MMCV [安装指南](https://mmcv.readthedocs.io/en/2.x/get_started/installation.html)。它需要你用指定 url 的形式手动指定对应的 PyTorch 和 CUDA 版本。

例如，以下命令安装为 PyTorch 1.10.x 和 CUDA 11.3 构建的 mmcv。

```shell
pip install 'mmcv>=2.0.0rc1' -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html
```

### 在 CPU 环境中安装

MMAction2 可以仅在 CPU 环境中安装。在 CPU 模式下，你可以完成训练、测试和模型推理等所有操作。

在 CPU 模式下，MMCV 的部分功能将不可用，通常是一些 GPU 编译的算子。不过不用担心， MMAction2 中几乎所有的模型都不会依赖这些算子。

### 通过Docker使用MMAction2

我们提供一个[Dockerfile](https://github.com/open-mmlab/mmaction2/blob/1.x/docker/Dockerfile)用来构建镜像，确保你的 [Docker版本](https://docs.docker.com/engine/install/)>=19.03.

```shell
# 例如构建PyTorch 1.6.0, CUDA 10.1, CUDNN 7的镜像
# 如果你喜欢其他版本,只要修改Dockerfile
docker build -f ./docker/Dockerfile --rm -t mmaction2 .
```

用以下命令运行 Docker 镜像：

```shell
docker run --gpus all --shm-size=8g -it -v {DATA_DIR}:/mmaction2/data mmaction2
```
