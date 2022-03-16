# Installation

We provide some tips for MMAction2 installation in this file.

<!-- TOC -->

- [Installation](#installation)
  - [Requirements](#requirements)
  - [Prepare environment](#prepare-environment)
  - [Install MMAction2](#install-mmaction2)
  - [Install with CPU only](#install-with-cpu-only)
  - [Another option: Docker Image](#another-option-docker-image)
  - [A from-scratch setup script](#a-from-scratch-setup-script)
  - [Developing with multiple MMAction2 versions](#developing-with-multiple-mmaction2-versions)
  - [Verification](#verification)

<!-- TOC -->

## Requirements

- Linux, Windows (We can successfully install mmaction2 on Windows and run inference, but we haven't tried training yet)
- Python 3.6+
- PyTorch 1.3+
- CUDA 9.2+ (If you build PyTorch from source, CUDA 9.0 is also compatible)
- GCC 5+
- [mmcv](https://github.com/open-mmlab/mmcv) 1.1.1+
- Numpy
- ffmpeg (4.2 is preferred)
- [decord](https://github.com/dmlc/decord) (optional, 0.4.1+): Install CPU version by `pip install decord==0.4.1` and install GPU version from source
- [PyAV](https://github.com/mikeboers/PyAV) (optional): `conda install av -c conda-forge -y`
- [PyTurboJPEG](https://github.com/lilohuang/PyTurboJPEG) (optional): `pip install PyTurboJPEG`
- [denseflow](https://github.com/open-mmlab/denseflow) (optional): See [here](https://github.com/innerlee/setup) for simple install scripts.
- [moviepy](https://zulko.github.io/moviepy/) (optional): `pip install moviepy`. See [here](https://zulko.github.io/moviepy/install.html) for official installation.  **Note**(according to [this issue](https://github.com/Zulko/moviepy/issues/693)) that:
  1. For Windows users, [ImageMagick](https://www.imagemagick.org/script/index.php) will not be automatically detected by MoviePy,
    there is a need to modify `moviepy/config_defaults.py` file by providing the path to the ImageMagick binary called `magick`, like `IMAGEMAGICK_BINARY = "C:\\Program Files\\ImageMagick_VERSION\\magick.exe"`
  2. For Linux users, there is a need to modify the `/etc/ImageMagick-6/policy.xml` file by commenting out
    `<policy domain="path" rights="none" pattern="@*" />` to `<!-- <policy domain="path" rights="none" pattern="@*" /> -->`, if [ImageMagick](https://www.imagemagick.org/script/index.php) is not detected by `moviepy`.
- [Pillow-SIMD](https://docs.fast.ai/performance.html#pillow-simd) (optional): Install it by the following scripts.

```shell
conda uninstall -y --force pillow pil jpeg libtiff libjpeg-turbo
pip   uninstall -y         pillow pil jpeg libtiff libjpeg-turbo
conda install -yc conda-forge libjpeg-turbo
CFLAGS="${CFLAGS} -mavx2" pip install --upgrade --no-cache-dir --force-reinstall --no-binary :all: --compile pillow-simd
conda install -y jpeg libtiff
```

:::{note}
You need to run `pip uninstall mmcv` first if you have mmcv installed.
If mmcv and mmcv-full are both installed, there will be `ModuleNotFoundError`.
:::

## Prepare environment

a. Create a conda virtual environment and activate it.

```shell
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab
```

b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), e.g.,

```shell
conda install pytorch torchvision -c pytorch
```

:::{note}
Make sure that your compilation CUDA version and runtime CUDA version match.
You can check the supported CUDA version for precompiled packages on the [PyTorch website](https://pytorch.org/).

`E.g.1` If you have CUDA 10.1 installed under `/usr/local/cuda` and would like to install PyTorch 1.5,
you need to install the prebuilt PyTorch with CUDA 10.1.

```shell
conda install pytorch cudatoolkit=10.1 torchvision -c pytorch
```

`E.g.2` If you have CUDA 9.2 installed under `/usr/local/cuda` and would like to install PyTorch 1.3.1.,
you need to install the prebuilt PyTorch with CUDA 9.2.

```shell
conda install pytorch=1.3.1 cudatoolkit=9.2 torchvision=0.4.2 -c pytorch
```

If you build PyTorch from source instead of installing the prebuilt package, you can use more CUDA versions such as 9.0.
:::

## Install MMAction2

We recommend you to install MMAction2 with [MIM](https://github.com/open-mmlab/mim).

```shell
pip install git+https://github.com/open-mmlab/mim.git
mim install mmaction2 -f https://github.com/open-mmlab/mmaction2.git
```

MIM can automatically install OpenMMLab projects and their requirements.

Or, you can install MMAction2 manually:

a. Install mmcv-full, we recommend you to install the pre-built package as below.

```shell
# pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.10.0/index.html
```

mmcv-full is only compiled on PyTorch 1.x.0 because the compatibility usually holds between 1.x.0 and 1.x.1. If your PyTorch version is 1.x.1, you can install mmcv-full compiled with PyTorch 1.x.0 and it usually works well.

```
# We can ignore the micro version of PyTorch
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.10/index.html
```

See [here](https://github.com/open-mmlab/mmcv#installation) for different versions of MMCV compatible to different PyTorch and CUDA versions.

Optionally you can choose to compile mmcv from source by the following command

```shell
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
MMCV_WITH_OPS=1 pip install -e .  # package mmcv-full, which contains cuda ops, will be installed after this step
# OR pip install -e .  # package mmcv, which contains no cuda ops, will be installed after this step
cd ..
```

Or directly run

```shell
pip install mmcv-full
# alternative: pip install mmcv
```

**Important:** You need to run `pip uninstall mmcv` first if you have mmcv installed. If mmcv and mmcv-full are both installed, there will be `ModuleNotFoundError`.

b. Clone the MMAction2 repository.

```shell
git clone https://github.com/open-mmlab/mmaction2.git
cd mmaction2
```

c. Install build requirements and then install MMAction2.

```shell
pip install -r requirements/build.txt
pip install -v -e .  # or "python setup.py develop"
```

If you build MMAction2 on macOS, replace the last command with

```shell
CC=clang CXX=clang++ CFLAGS='-stdlib=libc++' pip install -e .
```

d. Install mmdetection for spatial temporal detection tasks.

This part is **optional** if you're not going to do spatial temporal detection.

See [here](https://github.com/open-mmlab/mmdetection#installation) to install mmdetection.

:::{note}

1. The git commit id will be written to the version number with step b, e.g. 0.6.0+2e7045c. The version will also be saved in trained models.
   It is recommended that you run step b each time you pull some updates from github. If C++/CUDA codes are modified, then this step is compulsory.

2. Following the above instructions, MMAction2 is installed on `dev` mode, any local modifications made to the code will take effect without the need to reinstall it (unless you submit some commits and want to update the version number).

3. If you would like to use `opencv-python-headless` instead of `opencv-python`,
   you can install it before installing MMCV.

4. If you would like to use `PyAV`, you can install it with `conda install av -c conda-forge -y`.

5. Some dependencies are optional. Running `python setup.py develop` will only install the minimum runtime requirements.
   To use optional dependencies like `decord`, either install them with `pip install -r requirements/optional.txt`
   or specify desired extras when calling `pip` (e.g. `pip install -v -e .[optional]`,
   valid keys for the `[optional]` field are `all`, `tests`, `build`, and `optional`) like `pip install -v -e .[tests,build]`.

:::

## Install with CPU only

The code can be built for CPU only environment (where CUDA isn't available).

In CPU mode you can run the demo/demo.py for example.

## Another option: Docker Image

We provide a [Dockerfile](/docker/Dockerfile) to build an image.

```shell
# build an image with PyTorch 1.6.0, CUDA 10.1, CUDNN 7.
docker build -f ./docker/Dockerfile --rm -t mmaction2 .
```

**Important:** Make sure you've installed the [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).

Run it with command:

```shell
docker run --gpus all --shm-size=8g -it -v {DATA_DIR}:/mmaction2/data mmaction2
```

## A from-scratch setup script

Here is a full script for setting up MMAction2 with conda and link the dataset path (supposing that your Kinetics-400 dataset path is $KINETICS400_ROOT).

```shell
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab

# install latest pytorch prebuilt with the default prebuilt CUDA version (usually the latest)
conda install -c pytorch pytorch torchvision -y

# install the latest mmcv or mmcv-full, here we take mmcv as example
pip install mmcv

# install mmaction2
git clone https://github.com/open-mmlab/mmaction2.git
cd mmaction2
pip install -r requirements/build.txt
python setup.py develop

mkdir data
ln -s $KINETICS400_ROOT data
```

## Developing with multiple MMAction2 versions

The train and test scripts already modify the `PYTHONPATH` to ensure the script use the MMAction2 in the current directory.

To use the default MMAction2 installed in the environment rather than that you are working with, you can remove the following line in those scripts.

```shell
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
```

## Verification

To verify whether MMAction2 and the required environment are installed correctly,
we can run sample python codes to initialize a recognizer and inference a demo video:

```python
import torch
from mmaction.apis import init_recognizer, inference_recognizer

config_file = 'configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py'
device = 'cuda:0' # or 'cpu'
device = torch.device(device)

model = init_recognizer(config_file, device=device)
# inference the demo video
inference_recognizer(model, 'demo/demo.mp4')
```
