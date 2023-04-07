# MSG3D Project

[Disentangling and Unifying Graph Convolutions for Skeleton-Based Action Recognition](https://arxiv.org/abs/2003.14111)

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

Spatial-temporal graphs have been widely used by skeleton-based action recognition algorithms to model human action dynamics. To capture robust movement patterns from these graphs, long-range and multi-scale context aggregation and spatial-temporal dependency modeling are critical aspects of a powerful feature extractor. However, existing methods have limitations in achieving (1) unbiased long-range joint relationship modeling under multi-scale operators and (2) unobstructed cross-spacetime information flow for capturing complex spatial-temporal dependencies. In this work, we present (1) a simple method to disentangle multi-scale graph convolutions and (2) a unified spatial-temporal graph convolutional operator named G3D. The proposed multi-scale aggregation scheme disentangles the importance of nodes in different neighborhoods for effective long-range modeling. The proposed G3D module leverages dense cross-spacetime edges as skip connections for direct information propagation across the spatial-temporal graph. By coupling these proposals, we develop a powerful feature extractor named MS-G3D based on which our model outperforms previous state-of-the-art methods on three large-scale datasets: NTU RGB+D 60, NTU RGB+D 120, and Kinetics Skeleton 400.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/58767402/223127347-135bb92b-2dee-46d9-95fc-cebf65c27fc8.png" width="800"/>
</div>

## Usage

### Setup Environment

Please refer to [Get Started](https://mmaction2.readthedocs.io/en/latest/get_started.html) to install MMAction2.

At first, add the current folder to `PYTHONPATH`, so that Python can find your code. Run command in the current directory to add it.

> Please run it every time after you opened a new shell.

```shell
export PYTHONPATH=`pwd`:$PYTHONPATH
```

### Data Preparation

Prepare the NTU60 dataset according to the [instruction](https://github.com/open-mmlab/mmaction2/blob/1.x/tools/data/skeleton/README.md).

### Training commands

**To train with single GPU:**

```bash
mim train mmaction configs/msg3d_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py
```

**To train with multiple GPUs:**

```bash
mim train mmaction configs/msg3d_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py --launcher pytorch --gpus 8
```

**To train with multiple GPUs by slurm:**

```bash
mim train mmaction configs/msg3d_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py --launcher slurm \
    --gpus 8 --gpus-per-node 8 --partition $PARTITION
```

### Testing commands

**To test with single GPU:**

```bash
mim test mmaction configs/msg3d_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py --checkpoint $CHECKPOINT
```

**To test with multiple GPUs:**

```bash
mim test mmaction configs/msg3d_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py --checkpoint $CHECKPOINT --launcher pytorch --gpus 8
```

**To test with multiple GPUs by slurm:**

```bash
mim test mmaction configs/msg3d_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py --checkpoint $CHECKPOINT --launcher slurm \
    --gpus 8 --gpus-per-node 8 --partition $PARTITION
```

## Results

### NTU60_XSub_2D

| frame sampling strategy | modality | gpus | backbone | top1 acc | testing protocol |                     config                     |                     ckpt                     |                     log                     |
| :---------------------: | :------: | :--: | :------: | :------: | :--------------: | :--------------------------------------------: | :------------------------------------------: | :-----------------------------------------: |
|       uniform 100       |  joint   |  8   |  MSG3D   |   92.3   |     10 clips     | [config](./configs/msg3d_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/projects/msg3d/msg3d_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d/msg3d_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d_20230309-73b97296.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/projects/msg3d/msg3d_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d/msg3d_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.log) |

### NTU60_XSub_3D

| frame sampling strategy | modality | gpus | backbone | top1 acc | testing protocol |                     config                     |                     ckpt                     |                     log                     |
| :---------------------: | :------: | :--: | :------: | :------: | :--------------: | :--------------------------------------------: | :------------------------------------------: | :-----------------------------------------: |
|       uniform 100       |  joint   |  8   |  MSG3D   |   89.6   |     10 clips     | [config](./configs/msg3d_8xb16-joint-u100-80e_ntu60-xsub-keypoint-3d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/projects/msg3d/msg3d_8xb16-joint-u100-80e_ntu60-xsub-keypoint-3d/msg3d_8xb16-joint-u100-80e_ntu60-xsub-keypoint-3d_20230308-c325d222.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/projects/msg3d/msg3d_8xb16-joint-u100-80e_ntu60-xsub-keypoint-3d/msg3d_8xb16-joint-u100-80e_ntu60-xsub-keypoint-3d.log) |

## Citation

<!-- Replace to the citation of the paper your project refers to. -->

```bibtex
@inproceedings{liu2020disentangling,
  title={Disentangling and unifying graph convolutions for skeleton-based action recognition},
  author={Liu, Ziyu and Zhang, Hongwen and Chen, Zhenghao and Wang, Zhiyong and Ouyang, Wanli},
  booktitle={CVPR},
  pages={143--152},
  year={2020}
}
```

## Checklist

Here is a checklist of this project's progress, and you can ignore this part if you don't plan to contribute to MMAction2 projects.

- [x] Milestone 1: PR-ready, and acceptable to be one of the `projects/`.

  - [x] Finish the code

    <!-- The code's design shall follow existing interfaces and convention. For example, each model component should be registered into `mmaction.registry.MODELS` and configurable via a config file. -->

  - [x] Basic docstrings & proper citation

    <!-- Each major class should contains a docstring, describing its functionality and arguments. If your code is copied or modified from other open-source projects, don't forget to cite the source project in docstring and make sure your behavior is not against its license. Typically, we do not accept any code snippet under GPL license. [A Short Guide to Open Source Licenses](https://medium.com/nationwide-technology/a-short-guide-to-open-source-licenses-cf5b1c329edd) -->

  - [x] Converted checkpoint and results (Only for reproduction)

    <!-- If you are reproducing the result from a paper, make sure the model in the project can match that results. Also please provide checkpoint links or a checkpoint conversion script for others to get the pre-trained model. -->

- [x] Milestone 2: Indicates a successful model implementation.

  - [x] Training results

    <!-- If you are reproducing the result from a paper, train your model from scratch and verified that the final result can match the original result. Usually, ±0.1% is acceptable for the action recognition task on Kinetics400. -->

- [ ] Milestone 3: Good to be a part of our core package!

  - [ ] Unit tests

    <!-- Unit tests for the major module are required. [Example](https://github.com/open-mmlab/mmaction2/blob/1.x/tests/models/backbones/test_resnet.py) -->

  - [ ] Code style

    <!-- Refactor your code according to reviewer's comment. -->

  - [ ] `metafile.yml` and `README.md`

    <!-- It will used for MMAction2 to acquire your models. [Example](https://github.com/open-mmlab/mmaction2/blob/1.x/configs/recognition/swin/metafile.yml). In particular, you may have to refactor this README into a standard one. [Example](https://github.com/open-mmlab/mmaction2/blob/1.x/configs/recognition/swin/README.md) -->
