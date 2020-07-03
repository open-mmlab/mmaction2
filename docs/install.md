## Installation

### Requirements

- Linux (Windows is not officially supported)
- Python 3.6+
- PyTorch 1.3+
- CUDA 9.2+ (If you build PyTorch from source, CUDA 9.0 is also compatible)
- GCC 5+
- [mmcv](https://github.com/open-mmlab/mmcv) 0.5.7+
- Numpy
- ffmpeg (4.2 is preferred)
- [decord](https://github.com/dmlc/decord) (optional): install CPU version by `pip install decord` and install GPU version from source
- [PyAV](https://github.com/mikeboers/PyAV) (optional): `conda install av -c conda-forge -y`
- [PyTurboJPEG](https://github.com/lilohuang/PyTurboJPEG) (optional): `pip install PyTurboJPEG`
- [Pillow-SIMD](https://docs.fast.ai/performance.html#pillow-simd) (optional): install it by the following scripts.
```shell
conda uninstall -y --force pillow pil jpeg libtiff libjpeg-turbo
pip   uninstall -y         pillow pil jpeg libtiff libjpeg-turbo
conda install -yc conda-forge libjpeg-turbo
CFLAGS="${CFLAGS} -mavx2" pip install --upgrade --no-cache-dir --force-reinstall --no-binary :all: --compile pillow-simd
conda install -y jpeg libtiff
```

### Install mmaction

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

If you build PyTorch from source instead of installing the prebuilt pacakge, you can use more CUDA versions such as 9.0.

c. Clone the mmaction repository

```shell
git clone git@gitlab.sz.sensetime.com:open-mmlab/mmaction-lite.git
cd mmaction-lite
```

d. Install build requirements and then install mmaction

```shell
pip install -r requirements/build.txt
python setup.py develop
```

Note:

1. The git commit id will be written to the version number with step d, e.g. 0.6.0+2e7045c. The version will also be saved in trained models.
It is recommended that you run step d each time you pull some updates from github. If C++/CUDA codes are modified, then this step is compulsory.

2. Following the above instructions, mmaction is installed on `dev` mode, any local modifications made to the code will take effect without the need to reinstall it (unless you submit some commits and want to update the version number).

3. If you would like to use `opencv-python-headless` instead of `opencv-python`,
you can install it before installing MMCV.

4. If you would like to use `PyAV`, you can install it with `conda install av -c conda-forge -y`.

5. Some dependencies are optional. Running `python setup.py develop` will only install the minimum runtime requirements.
To use optional dependencies like `decord`, either install them with `pip install -r requirements/optional.txt`
or specify desired extras when calling `pip` (e.g. `pip install -v -e .[optional]`,
valid keys for the `[optional]` field are `all`, `tests`, `build`, and `optional`) like `pip install -v -e. optional`.

### Install with CPU only
The code can be built for CPU only environment (where CUDA isn't available).

In CPU mode you can run the demo/demo.py for example.

### Prepare datasets

It is recommended to symlink the dataset root to `$MMACTION/data`
If your folder structure is different, you may need to change the corresponding paths in config files.

```
mmaction
├── mmaction
├── tools
├── config
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

```

### A from-scratch setup script

Here is a full script for setting up mmaction with conda and link the dataset path (supposing that your Kinetics-400 dataset path is $KINETICS400_ROOT).

```shell
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab

conda install -c pytorch pytorch torchvision -y
git clone git@gitlab.sz.sensetime.com:open-mmlab/mmaction-lite.git
cd mmaction-lite
pip install -r requirements/build.txt
python setup.py develop

mkdir data
ln -s $KINETICS400_ROOT data
```

### Using multiple MMAction versions

The train and test scripts already modify the `PYTHONPATH` to ensure the script use the MMAction in the current directory.

To use the default MMAction installed in the environment rather than that you are working with, you can remove the following line in those scripts.

```shell
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
```
