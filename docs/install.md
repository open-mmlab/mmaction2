## Installation

### Requirements

- Linux (Windows is not officially supported)
- Python 3.6+
- PyTorch 1.3+
- CUDA 9.2+ (If you build PyTorch from source, CUDA 9.0 is also compatible)
- GCC 5+
- [mmcv](https://github.com/open-mmlab/mmcv) 1.1.1+
- Numpy
- ffmpeg (4.2 is preferred)
- [decord](https://github.com/dmlc/decord) (optional): Install CPU version by `pip install decord` and install GPU version from source
- [PyAV](https://github.com/mikeboers/PyAV) (optional): `conda install av -c conda-forge -y`
- [PyTurboJPEG](https://github.com/lilohuang/PyTurboJPEG) (optional): `pip install PyTurboJPEG`
- [denseflow](https://github.com/open-mmlab/denseflow) (optional): See [here](https://github.com/innerlee/setup) for simple install scripts.
- [moviepy](https://zulko.github.io/moviepy/) (optional): `pip install moviepy`. See [here](https://zulko.github.io/moviepy/install.html) for official installation. **Note**(according to [this issue](https://github.com/Zulko/moviepy/issues/693)) that:
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

### Install MMAction2

a. Create a conda virtual environment and activate it.

```shell
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab
```

b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), e.g.,

```shell
conda install pytorch torchvision -c pytorch
```

Note: Make sure that your compilation CUDA version and runtime CUDA version match.
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

c. Clone the mmaction2 repository

```shell
git clone https://github.com/open-mmlab/mmaction2.git
cd mmaction2
```

d. Install build requirements and then install mmaction2

```shell
pip install -r requirements/build.txt
pip install -v -e .  # or "python setup.py develop"
```

If you build mmaction2 on macOS, replace the last command with

```
CC=clang CXX=clang++ CFLAGS='-stdlib=libc++' pip install -e .
```


Note:

1. The git commit id will be written to the version number with step d, e.g. 0.6.0+2e7045c. The version will also be saved in trained models.
It is recommended that you run step d each time you pull some updates from github. If C++/CUDA codes are modified, then this step is compulsory.

2. Following the above instructions, mmaction2 is installed on `dev` mode, any local modifications made to the code will take effect without the need to reinstall it (unless you submit some commits and want to update the version number).

3. If you would like to use `opencv-python-headless` instead of `opencv-python`,
you can install it before installing MMCV.

4. If you would like to use `PyAV`, you can install it with `conda install av -c conda-forge -y`.

5. Some dependencies are optional. Running `python setup.py develop` will only install the minimum runtime requirements.
To use optional dependencies like `decord`, either install them with `pip install -r requirements/optional.txt`
or specify desired extras when calling `pip` (e.g. `pip install -v -e .[optional]`,
valid keys for the `[optional]` field are `all`, `tests`, `build`, and `optional`) like `pip install -v -e .[tests,build]`.

### Install with CPU only
The code can be built for CPU only environment (where CUDA isn't available).

In CPU mode you can run the demo/demo.py for example.

### Another option: Docker Image

We provide a [Dockerfile](/docker/Dockerfile) to build an image.

```shell
# build an image with PyTorch 1.5, CUDA 10.1
docker build -t mmaction docker/
```

Run it with

```shell
docker run --gpus all --shm-size=8g -it -v {DATA_DIR}:/mmaction/data mmaction
```

### A from-scratch setup script

Here is a full script for setting up mmaction2 with conda and link the dataset path (supposing that your Kinetics-400 dataset path is $KINETICS400_ROOT).

```shell
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab

conda install -c pytorch pytorch torchvision -y
git clone https://github.com/open-mmlab/mmaction.git
cd mmaction
pip install -r requirements/build.txt
python setup.py develop

mkdir data
ln -s $KINETICS400_ROOT data
```

### Using multiple MMAction2 versions

The train and test scripts already modify the `PYTHONPATH` to ensure the script use the MMAction2 in the current directory.

To use the default MMAction2 installed in the environment rather than that you are working with, you can remove the following line in those scripts.

```shell
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
```
