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

Please refer to [Installation](https://mmaction2.readthedocs.io/en/latest/get_started/installation.html) to install MMAction2.

Assume that you are located at `$MMACTION2/projects/ctrgcn`.

Add the current folder to `PYTHONPATH`, so that Python can find your code. Run the following command in the current directory to add it.

> Please run it every time after you opened a new shell.

```shell
export PYTHONPATH=`pwd`:$PYTHONPATH
```

### Data Preparation

Prepare the NTU60 dataset according to the [instruction](https://github.com/open-mmlab/mmaction2/blob/main/tools/data/skeleton/README.md).

Create a symbolic link from `$MMACTION2/data` to `./data` in the current directory, so that Python can locate your data. Run the following command in the current directory to create the symbolic link.

```shell
ln -s ../../data ./data
```

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
