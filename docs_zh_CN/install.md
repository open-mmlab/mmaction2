# 安装

本文档提供了安装 MMAction2 的相关步骤。

<!-- TOC -->

- [安装依赖包](#安装依赖包)
- [准备环境](#准备环境)
- [MMAction2 的安装步骤](#MMAction2-的安装步骤)
- [CPU 环境下的安装步骤](#CPU-环境下的安装步骤)
- [利用 Docker 镜像安装 MMAction2](#利用-Docker-镜像安装-MMAction2)
- [源码安装 MMAction2](#源码安装-MMAction2)
- [在多个 MMAction2 版本下进行开发](#在多个-MMAction2-版本下进行开发)
- [安装验证](#安装验证)

<!-- TOC -->

## 安装依赖包

- Linux (Windows 系统暂未有官方支持)
- Python 3.6+
- PyTorch 1.3+
- CUDA 9.2+ (如果要从源码对 PyTorch 进行编译, CUDA 9.0 版本同样可以兼容)
- GCC 5+
- [mmcv](https://github.com/open-mmlab/mmcv) 1.1.1+
- Numpy
- ffmpeg (4.2 版本最佳)
- [decord](https://github.com/dmlc/decord) (可选项, 0.4.1+)：使用 `pip install decord==0.4.1` 命令安装其 CPU 版本，GPU 版本需从源码进行编译。
- [PyAV](https://github.com/mikeboers/PyAV) (可选项)：`conda install av -c conda-forge -y`。
- [PyTurboJPEG](https://github.com/lilohuang/PyTurboJPEG) (可选项)：`pip install PyTurboJPEG`。
- [denseflow](https://github.com/open-mmlab/denseflow) (可选项)：可参考 [这里](https://github.com/innerlee/setup) 获取简便安装步骤。
- [moviepy](https://zulko.github.io/moviepy/) (可选项)：`pip install moviepy`. 官方安装步骤可参考 [这里](https://zulko.github.io/moviepy/install.html)。**特别地**，如果安装过程碰到 [这个问题](https://github.com/Zulko/moviepy/issues/693)，可参考：
    1. 对于 Windows 用户, [ImageMagick](https://www.imagemagick.org/script/index.php) 将不会被 MoviePy 自动检测到，
    用户需要对 `moviepy/config_defaults.py` 文件进行修改，以提供 ImageMagick 的二进制文件（即，`magick`）的路径，如 `IMAGEMAGICK_BINARY = "C:\\Program Files\\ImageMagick_VERSION\\magick.exe"`
    2. 对于 Linux 用户, 如果 [ImageMagick](https://www.imagemagick.org/script/index.php) 没有被 `moviepy` 检测到，用于需要对 `/etc/ImageMagick-6/policy.xml` 文件进行修改，把文件中的
    `<policy domain="path" rights="none" pattern="@*" />` 代码行修改为 `<!-- <policy domain="path" rights="none" pattern="@*" /> -->`。
- [Pillow-SIMD](https://docs.fast.ai/performance.html#pillow-simd) (可选项)：可使用如下脚本进行安装：

```shell
conda uninstall -y --force pillow pil jpeg libtiff libjpeg-turbo
pip   uninstall -y         pillow pil jpeg libtiff libjpeg-turbo
conda install -yc conda-forge libjpeg-turbo
CFLAGS="${CFLAGS} -mavx2" pip install --upgrade --no-cache-dir --force-reinstall --no-binary :all: --compile pillow-simd
conda install -y jpeg libtiff
```

**注意**：用户需要首先运行 `pip uninstall mmcv` 命令，以确保 mmcv 被成功安装。
如果 mmcv 和 mmcv-full 同时被安装, 会报 `ModuleNotFoundError` 的错误。

## 准备环境

a. 创建并激活 conda 虚拟环境，如：

```shell
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab
```

b. 根据 [官方文档](https://pytorch.org/) 进行 PyTorch 和 torchvision 的安装，如：

```shell
conda install pytorch torchvision -c pytorch
```

**注**：确保 CUDA 的编译版本和 CUDA 的运行版本相匹配。
用户可以参照 [PyTorch 官网](https://pytorch.org/) 对预编译包所支持的 CUDA 版本进行核对。

`例 1`：如果用户的 `/usr/local/cuda` 文件夹下已安装 CUDA 10.1 版本，并且想要安装 PyTorch 1.5 版本，
则需要安装 CUDA 10.1 下预编译的 PyTorch。

```shell
conda install pytorch cudatoolkit=10.1 torchvision -c pytorch
```

`例 2`：如果用户的 `/usr/local/cuda` 文件夹下已安装 CUDA 9.2 版本，并且想要安装 PyTorch 1.3.1 版本，
则需要安装 CUDA 9.2 下预编译的 PyTorch。

```shell
conda install pytorch=1.3.1 cudatoolkit=9.2 torchvision=0.4.2 -c pytorch
```

如果 PyTorch 是由源码进行编译安装（而非直接下载预编译好的安装包），则可以使用更多的 CUDA 版本（如 9.0 版本）。

## MMAction2 的安装步骤

这里推荐用户使用 [MIM](https://github.com/open-mmlab/mim) 安装 MMAction2。

```shell
pip install git+https://github.com/open-mmlab/mim.git
mim install mmaction2
```

MIM 可以自动安装 OpenMMLab 项目及其依赖。

或者，用户也可以通过以下步骤手动安装 MMAction2。

a. 安装 mmcv。MMAction2 推荐用户使用如下的命令安装预编译好的 mmcv。

```shell
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
```

其中，命令里 url 的 ``{cu_version}`` 和 ``{torch_version}`` 变量需由用户进行指定。
例如，如果想要安装 ``CUDA 11`` 和 ``PyTorch 1.7.0`` 下的最新版 ``mmcv-full``，可使用以下命令：

```shell
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html
```

可查阅 [这里](https://github.com/open-mmlab/mmcv#installation) 以参考不同版本的 MMCV 所兼容的 PyTorch 和 CUDA 版本。

另外，用户也可以通过使用以下命令从源码进行编译：

```shell
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
MMCV_WITH_OPS=1 pip install -e .  # mmcv-full 包含一些 cuda 算子，执行该步骤会安装 mmcv-full（而非 mmcv）
# 或者使用 pip install -e .  # 这个命令安装的 mmcv 将不包含 cuda ops，通常适配 CPU（无 GPU）环境
cd ..
```

或者直接运行脚本：

```shell
pip install mmcv-full
```

**注意**：如果 mmcv 已经被安装，用户需要使用 `pip uninstall mmcv` 命令进行卸载。如果 mmcv 和 mmcv-full 同时被安装, 会报 `ModuleNotFoundError` 的错误。

b. 克隆 MMAction2 库。

```shell
git clone https://github.com/open-mmlab/mmaction2.git
cd mmaction2
```

c. 安装依赖包和 MMAction2。

```shell
pip install -r requirements/build.txt
pip install -v -e .  # or "python setup.py develop"
```

如果是在 macOS 环境安装 MMAction2，则需使用如下命令：

```shell
CC=clang CXX=clang++ CFLAGS='-stdlib=libc++' pip install -e .
```

d. 安装 mmdetection 以支持时空检测任务。

如果用户不想做时空检测相关任务，这部分步骤可以选择跳过。

可参考 [这里](https://github.com/open-mmlab/mmdetection#installation) 进行 mmdetection 的安装。

注意：

1. 在步骤 b 中，git commit 的 id 将会被写到版本号中，如 0.6.0+2e7045c。这个版本号也会被保存到训练好的模型中。
   这里推荐用户每次在步骤 b 中对本地代码和 github 上的源码进行同步。如果 C++/CUDA 代码被修改，就必须进行这一步骤。

2. 根据上述步骤，MMAction2 就会以 `dev` 模式被安装，任何本地的代码修改都会立刻生效，不需要再重新安装一遍（除非用户提交了 commits，并且想更新版本号）。

3. 如果用户想使用 `opencv-python-headless` 而不是 `opencv-python`，可再安装 MMCV 前安装 `opencv-python-headless`。

4. 如果用户想使用 `PyAV`，可以通过 `conda install av -c conda-forge -y` 进行安装。

5. 一些依赖包是可选的。运行 `python setup.py develop` 将只会安装运行代码所需的最小要求依赖包。
   要想使用一些可选的依赖包，如 `decord`，用户需要通过 `pip install -r requirements/optional.txt` 进行安装，
   或者通过调用 `pip`（如 `pip install -v -e .[optional]`，这里的 `[optional]` 可替换为 `all`，`tests`，`build` 或 `optional`） 指定安装对应的依赖包，如 `pip install -v -e .[tests,build]`。

## CPU 环境下的安装步骤

MMAction2 可以在只有 CPU 的环境下安装（即无法使用 GPU 的环境）。

在 CPU 模式下，用户可以运行 `demo/demo.py` 的代码。

## 利用 Docker 镜像安装 MMAction2

MMAction2 提供一个 [Dockerfile](/docker/Dockerfile) 用户创建 docker 镜像。

```shell
# 创建拥有 PyTorch 1.6.0, CUDA 10.1, CUDNN 7 配置的 docker 镜像.
docker build -f ./docker/Dockerfile --rm -t mmaction2 .
```

**注意**：用户需要确保已经安装了 [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)。

运行以下命令：

```shell
docker run --gpus all --shm-size=8g -it -v {DATA_DIR}:/mmaction2/data mmaction2
```

## 源码安装 MMAction2

这里提供了 conda 下安装 MMAction2 并链接数据集路径的完整脚本（假设 Kinetics-400 数据的路径在 $KINETICS400_ROOT）。

```shell
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab

# 安装最新的，使用默认版本的 CUDA 版本（一般为最新版本）预编译的 PyTorch 包
conda install -c pytorch pytorch torchvision -y

# 安装最新版本的 mmcv 或 mmcv-full，这里以 mmcv 为例
pip install mmcv

# 安装 mmaction2
git clone https://github.com/open-mmlab/mmaction2.git
cd mmaction2
pip install -r requirements/build.txt
python setup.py develop

mkdir data
ln -s $KINETICS400_ROOT data
```

## 在多个 MMAction2 版本下进行开发

MMAction2 的训练和测试脚本已经修改了 `PYTHONPATH` 变量，以确保其能够运行当前目录下的 MMAction2。

如果想要运行环境下默认的 MMAction2，用户需要在训练和测试脚本中去除这一行：

```shell
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
```

## 安装验证

为了验证 MMAction2 和所需的依赖包是否已经安装成功，
用户可以运行以下的 python 代码，以测试其是否能成功地初始化动作识别器，并进行演示视频的推理：

```python
import torch
from mmaction.apis import init_recognizer, inference_recognizer

config_file = 'configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py'
device = 'cuda:0' # 或 'cpu'
device = torch.device(device)

model = init_recognizer(config_file, device=device)
# 进行演示视频的推理
inference_recognizer(model, 'demo/demo.mp4', 'demo/label_map_k400.txt')
```
