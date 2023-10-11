# Installation

## Prerequisites

In this section we demonstrate how to prepare an environment with PyTorch.

MMAction2 works on Linux, Windows and macOS. It requires Python 3.7+, CUDA 10.2+ and PyTorch 1.8+.

```{note}
If you are experienced with PyTorch and have already installed it, just skip this part and jump to the [next section](#installation). Otherwise, you can follow these steps for the preparation.
```

**Step 1.** Download and install Miniconda from the [official website](https://docs.conda.io/en/latest/miniconda.html).

**Step 2.** Create a conda environment and activate it.

```shell
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
```

**Step 3.** Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/), e.g.

On GPU platforms:

```shell
conda install pytorch torchvision -c pytorch
```

```{warning}
This command will automatically install the latest version PyTorch and cudatoolkit, please check whether they match your environment.
```

On CPU platforms:

```shell
conda install pytorch torchvision cpuonly -c pytorch
```

## Best Practices

We recommend that users follow our best practices to install MMAction2. However, the whole process is highly customizable. See [Customize Installation](#customize-installation) section for more information.

**Step 1.** Install [MMEngine](https://github.com/open-mmlab/mmengine), [MMCV](https://github.com/open-mmlab/mmcv), [MMDetection](https://github.com/open-mmlab/mmdetection) (optional) and [MMPose](https://github.com/open-mmlab/mmpose) (optional) using [MIM](https://github.com/open-mmlab/mim).

```shell
pip install -U openmim
mim install mmengine
mim install mmcv
mim install mmdet
mim install mmpose
```

**Step 2.** Install MMAction2.

According to your needs, we support two install modes:

- [Install from source (Recommended)](#build-mmaction2-from-source): You want to develop your own action recognition task or new features on MMAction2 framework. For example, adding new dataset or new models. Thus, you can use all tools we provided.
- [Install as a Python package](#install-as-a-python-package): You just want to call MMAction2's APIs or import MMAction2's modules in your project.

### Build MMAction2 from source

In this case, install mmaction2 from source:

```shell
git clone https://github.com/open-mmlab/mmaction2.git
cd mmaction2
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without re-installation.
```

Optionally, if you want to contribute to MMAction2 or experience experimental functions, please checkout to the `dev-1.x` branch:

```shell
git checkout dev-1.x
```

### Install as a Python package

Just install with pip.

```shell
pip install mmaction2
```

## Verify the installation

To verify whether MMAction2 is installed correctly, we provide some sample codes to run an inference demo.

**Step 1.** Download the config and checkpoint files.

```shell
mim download mmaction2 --config tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb --dest .
```

**Step 2.** Verify the inference demo.

Option (a). If you install mmaction2 from source, you can run the following command:

```shell
# The demo.mp4 and label_map_k400.txt are both from Kinetics-400
python demo/demo.py tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb.py \
    tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb_20220906-2692d16c.pth \
    demo/demo.mp4 tools/data/kinetics/label_map_k400.txt
```

You will see the top-5 labels with corresponding scores in your terminal.

Option (b). If you install mmaction2 as a python package, you can run the following codes in your python interpreter, which will do the similar verification:

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

## Customize Installation

### CUDA versions

When installing PyTorch, you may need to specify the version of CUDA. If you are
not clear on which to choose, follow our recommendations:

- For Ampere-based NVIDIA GPUs, such as GeForce 30 series and NVIDIA A100, CUDA 11 is a must.
- For older NVIDIA GPUs, CUDA 11 is backward compatible, but CUDA 10.2 offers better compatibility and is more lightweight.

Please make sure the GPU driver satisfies the minimum version requirements. See [this table](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions__table-cuda-toolkit-driver-versions) for more information.

```{note}
Installing CUDA runtime libraries is enough if you follow our best practices,
because no CUDA code will be compiled locally. However if you hope to compile
MMCV from source or develop other CUDA operators, you need to install the
complete CUDA toolkit from NVIDIA's [website](https://developer.nvidia.com/cuda-downloads),
and its version should match the CUDA version of PyTorch. i.e., the specified
version of cudatoolkit in `conda install` command.
```

### Install MMCV without MIM

MMCV contains C++ and CUDA extensions, so it depends on PyTorch in a complex
way. MIM solves such dependencies automatically and makes the installation
easier. However, it is not a must.

To install MMCV with pip instead of MIM, please follow
[MMCV installation guides](https://mmcv.readthedocs.io/en/latest/get_started/installation.html).
This requires manually specifying a find-url based on PyTorch version and its CUDA version.

For example, the following command install mmcv built for PyTorch 1.10.x and CUDA 11.3.

```shell
pip install mmcv -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html
```

### Install on CPU-only platforms

MMAction2 can be built for CPU-only environment. In CPU mode you can train, test or inference a model.

Some functionalities are gone in this mode, usually GPU-compiled ops. But don't
worry, almost all models in MMAction2 don't depend on these ops.

### Using MMAction2 with Docker

We provide a [Dockerfile](https://github.com/open-mmlab/mmaction2/blob/main/docker/Dockerfile)
to build an image. Ensure that your [docker version](https://docs.docker.com/engine/install/) >=19.03.

```shell
# build an image with PyTorch 1.8.1, CUDA 10.2, CUDNN 7.
# If you prefer other versions, just modified the Dockerfile
docker build -f ./docker/Dockerfile --rm -t mmaction2 .
```

Run it with

```shell
docker run --gpus all --shm-size=8g -it -v {DATA_DIR}:/mmaction2/data mmaction2
```

## Troubleshooting

1. When migrating from the old version `0.x` to the new version `1.x`, you may encounter issues with mismatched versions of dependent libraries. Below is a display of the versions of each dependent library after following the aforementioned installation process, as shown by `pip list` command. Please ensure that the versions of each dependent library displayed in your terminal are greater than or equal to (i.e., `>=`) the versions shown below for each dependent library.

```shell
mmaction2                1.0.0
mmcv                     2.0.0
mmdet                    3.0.0
mmengine                 0.7.2
mmpose                   1.0.0
```
