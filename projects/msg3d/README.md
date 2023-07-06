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

Please refer to [Installation](https://mmaction2.readthedocs.io/en/latest/get_started/installation.html) to install MMAction2.

Assume that you are located at `$MMACTION2/projects/msg3d`.

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
