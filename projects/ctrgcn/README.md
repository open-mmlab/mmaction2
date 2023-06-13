# CTRGCN Project

[Channel-wise Topology Refinement Graph Convolution for Skeleton-Based Action Recognition](https://arxiv.org/abs/2107.12213)

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

Graph convolutional networks (GCNs) have been widely used and achieved remarkable results in skeleton-based action recognition. In GCNs, graph topology dominates feature aggregation and therefore is the key to extracting representative features. In this work, we propose a novel Channel-wise Topology Refinement Graph Convolution (CTR-GC) to dynamically learn different topologies and effectively aggregate joint features in different channels for skeleton-based action recognition. The proposed CTR-GC models channel-wise topologies through learning a shared topology as a generic prior for all channels and refining it with channel-specific correlations for each channel. Our refinement method introduces few extra parameters and significantly reduces the difficulty of modeling channel-wise topologies. Furthermore, via reformulating graph convolutions into a unified form, we find that CTR-GC relaxes strict constraints of graph convolutions, leading to stronger representation capability. Combining CTR-GC with temporal modeling modules, we develop a powerful graph convolutional network named CTR-GCN which notably outperforms state-of-the-art methods on the NTU RGB+D, NTU RGB+D 120, and NW-UCLA datasets.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/58767402/223147561-9158fd51-8963-47c9-9338-de70470820cc.png" width="800"/>
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
mim train mmaction configs/ctrgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py
```

**To train with multiple GPUs:**

```bash
mim train mmaction configs/ctrgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py --launcher pytorch --gpus 8
```

**To train with multiple GPUs by slurm:**

```bash
mim train mmaction configs/ctrgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py --launcher slurm \
    --gpus 8 --gpus-per-node 8 --partition $PARTITION
```

### Testing commands

**To test with single GPU:**

```bash
mim test mmaction configs/ctrgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py --checkpoint $CHECKPOINT
```

**To test with multiple GPUs:**

```bash
mim test mmaction configs/ctrgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py --checkpoint $CHECKPOINT --launcher pytorch --gpus 8
```

**To test with multiple GPUs by slurm:**

```bash
mim test mmaction configs/ctrgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py --checkpoint $CHECKPOINT --launcher slurm \
    --gpus 8 --gpus-per-node 8 --partition $PARTITION
```

## Results

### NTU60_XSub_2D

| frame sampling strategy | modality | gpus | backbone | top1 acc | testing protocol |                     config                     |                     ckpt                     |                     log                     |
| :---------------------: | :------: | :--: | :------: | :------: | :--------------: | :--------------------------------------------: | :------------------------------------------: | :-----------------------------------------: |
|       uniform 100       |  joint   |  8   |  CTRGCN  |   89.6   |     10 clips     | [config](./configs/ctrgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/projects/ctrgcn/ctrgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d/ctrgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d_20230308-7aba454e.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/projects/ctrgcn/ctrgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d/ctrgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.log) |

### NTU60_XSub_3D

| frame sampling strategy | modality | gpus | backbone | top1 acc | testing protocol |                     config                     |                     ckpt                     |                     log                     |
| :---------------------: | :------: | :--: | :------: | :------: | :--------------: | :--------------------------------------------: | :------------------------------------------: | :-----------------------------------------: |
|       uniform 100       |  joint   |  8   |  CTRGCN  |   89.0   |     10 clips     | [config](./configs/ctrgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-3d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/projects/ctrgcn/ctrgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-3d/ctrgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-3d_20230308-950dca0a.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/projects/ctrgcn/ctrgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-3d/ctrgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-3d.log) |

## Citation

<!-- Replace to the citation of the paper your project refers to. -->

```bibtex
@inproceedings{chen2021channel,
  title={Channel-wise topology refinement graph convolution for skeleton-based action recognition},
  author={Chen, Yuxin and Zhang, Ziqi and Yuan, Chunfeng and Li, Bing and Deng, Ying and Hu, Weiming},
  booktitle={CVPR},
  pages={13359--13368},
  year={2021}
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
