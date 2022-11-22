# 前置条件

在这节，我们阐述了如何准备PyTorch环境。

MMAction2运行在Linux、Windows和MacOS，它需要Python 3.6+，CUDA 9.2+ 和 PyTorch 1.6+。

```
如果你熟悉PyTorch并且已经安装了它，请跳过改章节至[下一节](#安装)
```

\*\*第一步：\*\*从[官网](https://docs.conda.io/en/latest/miniconda.html)下载和安装Miniconda

\*\*第二步：\*\*创建conda环境，并激活它

```shell
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
```

\*\*第三步：\*\*根据[官方指示](https://pytorch.org/get-started/locally/)安装PyTorch

在GPU平台：

```shell
conda install pytorch torchvision -c pytorch
```

```
这条命令将自动安装最新版本的PyTorch和cudatoolkit,请检查是否和你的环境匹配
```

在CPU平台：

```shell
conda install pytorch torchvision cpuonly -c pytorch
```

# 安装

我们推荐用户跟随我们的最佳实践安装 MMAction2 .但是整个过程是高度自定义的，见[自定义安装](#自定义安装)章节获取更多信息

## 最佳实践

\*\*第一步：\*\*使用 MIM 安装 MMEngine 和 MMCV

```shell
pip install -U openmim
mim install mmengine 'mmcv>=2.0.0rc1'
```

\*\*第二步：\*\*安装 MMAction2

根据你的需要，我们支持两种安装模式

- [从源码安装（推荐）](#从源码安装)：你想开发你自己的动作识别人物或者在 MMAction2 上开发新功能，例如，添加新的数据集或者新的模型。因此，你可以使用我们提供的所有工具。
- [作为Python包安装](<>)：你只想调用 MMAction2 的API或者在你的项目中导入 MMAction2 的模块

### 从源码安装

这种方案中，从源码安装 MMAction2 ：

```shell
git clone https://github.com/open-mmlab/mmaction2.git
cd mmaction2
git checkout 1.x
pip install -v -e .
# "-v" 表示 verbose ,更详细的输出
# "-e" 表示以可编辑的模式安装一个项目
# 因此任何本地源码的修改将无需重新安装就可以生效
```

可选，如果你想为 MMAction2 贡献代码，或者体验试验中的功能，请切换到`dev-1.x`分支

```shell
git checkout dev-1.x
```

### 作为Python包安装

只要用pip安装

```shell
pip install "mmaction2>=1.0rc0"
```

## 验证安装

为验证 MMAction2 是否被正确安装，我们提供了一些简单的代码用于运行一个推理demo

\*\*第一步：\*\*我们需要下载config配置文件和checkpoint模型断点文件

```shell
mim download mmaction2 --config tsn_r50_8xb32-1x1x8-100e_kinetics400-rgb --dest .
```

\*\*第二步：\*\*验证推理demo

选项(a)：如果你从源码安装 MMAction2，只要运行下面的指令：

```shell
# demo.mp4 和 label_map_k400.txt 都来自于 Kinetics-400
python demo/demo.py tsn_r50_8xb32-1x1x8-100e_kinetics400-rgb.py \
    tsn_r50_8xb32-1x1x8-100e_kinetics400-rgb_20220818-2692d16c.pth \
    demo/demo.mp4 tools/data/kinetics/label_map_k400.txt
```

你将在终端看到5个最高相关分数的label

选项(b)：如果你通过pip方式安装 MMAction2，打开你的Python解释器，复制并粘贴以下代码：

```python
from mmaction.apis import init_recognizer, inference_recognizer
from mmaction.utils import register_all_modules

config_file = 'tsn_r50_8xb32-1x1x8-100e_kinetics400-rgb.py'
checkpoint_file = 'tsn_r50_8xb32-1x1x8-100e_kinetics400-rgb_20220818-2692d16c.pth'
video_file = 'demo/demo.mp4'
label_file = 'tools/data/kinetics/label_map_k400.txt'
register_all_modules()  # register all modules and set mmaction2 as the default scope.
model = init_recognizer(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'
results = inference_recognizer(model, video_file)

labels = open(label_file).readlines()
labels = [x.strip() for x in labels]
results = [(labels[k[0]], k[1]) for k in results]
print('The top-5 labels with corresponding scores are:')
for result in results:
    print(f'{result[0]}: ', result[1])
```

## 自定义安装

### CUDA 版本

安装PyTorch时，你需要提供特定的CUDA的版本。如果你不清楚选择哪个版本，跟着我们的推荐：

- 对于Ampere-based NVIDIA GPUs，例如GeForce 30系列和NVIDIA A100，CUDA 11是必须的
- 对于旧的NVIDIA GPUs，CUDA 11是向下兼容的，但是CUDA 10.2 提供更好的兼容性而且更加轻量

请确保GPU驱动满足要求的最低版本，见[此表格](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions__table-cuda-toolkit-driver-versions)获得更多信息。

```
如果你跟着我们的最佳实践，安装CUDA runtime libraries就足够了，因为没有CUDA代码会在本地被编译。但是如果你希望从源码编译 MMCV 或者开发其他CUDA算子，你需要从NVIDIA的[官网](https://developer.nvidia.com/cuda-downloads)安装完整的CUDA toolkit，而且它的版本需要和PyTorch的CUDA版本相匹配，也就是在`conda install`指令中cudatoolkit的制定版本
```

### 不使用 MIM 安装 MMCV

MMCV 包含 C++ 和 CUDA 扩展，因此以复杂的方式依赖于 PyTorch。 MIM 会自动解决此类依赖关系并使安装更容易。但是，这不是必须的。

要使用 pip 而不是 MIM 安装 MMCV，请遵循 MMCV 安装指南。这需要根据 PyTorch 版本及其 CUDA 版本手动指定 find-url。

例如，以下命令安装为 PyTorch 1.10.x 和 CUDA 11.3 构建的 mmcv。

```shell
pip install 'mmcv>=2.0.0rc1' -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html
```

### 在仅支持CPU的平台上安装

MMAction2 可以为只有 CPU 的环境构建。在 CPU 模式下，你可以训练、测试或推理模型。

一些功能在这种模式下消失了，通常是 GPU 编译的操作。但别担心，MMAction2 中的几乎所有模型都不依赖于这些操作。
