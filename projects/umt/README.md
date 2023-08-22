# UMT Project

[Unmasked Teacher: Towards Training-Efficient Video Foundation Models](https://arxiv.org/abs/2303.16058)

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

Video Foundation Models (VFMs) have received limited exploration due to high computational costs and data scarcity. Previous VFMs rely on Image Foundation Models (IFMs), which face challenges in transferring to the video domain. Although VideoMAE has trained a robust ViT from limited data, its low-level reconstruction poses convergence difficulties and conflicts with high-level cross-modal alignment. This paper proposes a training-efficient method for temporal-sensitive VFMs that integrates the benefits of existing methods. To increase data efficiency, we mask out most of the low-semantics video tokens, but selectively align the unmasked tokens with IFM, which serves as the UnMasked Teacher (UMT). By providing semantic guidance, our method enables faster convergence and multimodal friendliness. With a progressive pre-training framework, our model can handle various tasks including scene-related, temporal-related, and complex video-language understanding. Using only public sources for pre-training in 6 days on 32 A100 GPUs, our scratch-built ViT-L/16 achieves state-of-the-art performances on various video tasks.

<!-- [IMAGE] -->

<div align=center>
<img src="https://github-production-user-asset-6210df.s3.amazonaws.com/58767402/262291190-bdaa6899-e1d6-460f-b329-23d8b38511f3.png" width="800"/>
</div>

## Usage

### Setup Environment

Please refer to [Installation](https://mmaction2.readthedocs.io/en/latest/get_started/installation.html) to install MMAction2.

Assume that you are located at `$MMACTION2/projects/umt`.

Add the current folder to `PYTHONPATH`, so that Python can find your code. Run the following command in the current directory to add it.

> Please run it every time after you opened a new shell.

```shell
export PYTHONPATH=`pwd`:$PYTHONPATH
```

### Data Preparation

Prepare the Kinetics dataset according to the [instruction](https://github.com/open-mmlab/mmaction2/tree/main/tools/data/kinetics#readme).

Create a symbolic link from `$MMACTION2/data` to `./data` in the current directory, so that Python can locate your data. Run the following command in the current directory to create the symbolic link.

```shell
ln -s ../../data ./data
```

### Testing commands

**To test with single GPU:**

```bash
mim test mmaction configs/umt-base-p16-res224_kinetics710-pre-ft_u8_k400-rgb.py --checkpoint $CHECKPOINT
```

**To test with multiple GPUs:**

```bash
mim test mmaction configs/umt-base-p16-res224_kinetics710-pre-ft_u8_k400-rgb.py --checkpoint $CHECKPOINT --launcher pytorch --gpus 8
```

**To test with multiple GPUs by slurm:**

```bash
mim test mmaction configs/umt-base-p16-res224_kinetics710-pre-ft_u8_k400-rgb.py --checkpoint $CHECKPOINT --launcher slurm \
    --gpus 8 --gpus-per-node 8 --partition $PARTITION
```

## Results

### Kinetics400

| frame sampling strategy | resolution | backbone |  pretrain   | top1 acc | testing protocol |                             config                              |                             ckpt                              |
| :---------------------: | :--------: | :------: | :---------: | :------: | :--------------: | :-------------------------------------------------------------: | :-----------------------------------------------------------: |
|        uniform 8        |  224x224   |  UMT-B   | Kinetics710 |  87.33   | 4 clips x 3 crop | [config](./configs/umt-base-p16-res224_kinetics710-pre-ft_u8_k400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/projects/umt/umt-base-p16-res224_kinetics710-pre-ft_u8_k400-rgb/umt-base-p16-res224_kinetics710-pre-ft_u8_k400-rgb.pth) |
|        uniform 8        |  224x224   |  UMT-L   | Kinetics710 |  90.21   | 4 clips x 3 crop | [config](./configs/umt-large-p16-res224_kinetics710-pre-ft_u8_k400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/projects/umt/umt-large-p16-res224_kinetics710-pre-ft_u8_k400-rgb/umt-large-p16-res224_kinetics710-pre-ft_u8_k400-rgb.pth) |

### Kinetics700

| frame sampling strategy | resolution | backbone |  pretrain   | top1 acc | testing protocol |                             config                              |                             ckpt                              |
| :---------------------: | :--------: | :------: | :---------: | :------: | :--------------: | :-------------------------------------------------------------: | :-----------------------------------------------------------: |
|        uniform 8        |  224x224   |  UMT-B   | Kinetics710 |  77.95   | 4 clips x 3 crop | [config](./configs/umt-base-p16-res224_kinetics710-pre-ft_u8_k700-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/projects/umt/umt-base-p16-res224_kinetics710-pre-ft_u8_k700-rgb/umt-base-p16-res224_kinetics710-pre-ft_u8_k700-rgb.pth) |
|        uniform 8        |  224x224   |  UMT-L   | Kinetics710 |  82.79   | 4 clips x 3 crop | [config](./configs/umt-large-p16-res224_kinetics710-pre-ft_u8_k700-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/projects/umt/umt-large-p16-res224_kinetics710-pre-ft_u8_k700-rgb/umt-large-p16-res224_kinetics710-pre-ft_u8_k700-rgb.pth) |

## Citation

<!-- Replace to the citation of the paper your project refers to. -->

```bibtex
@article{li2023unmasked,
  title={Unmasked teacher: Towards training-efficient video foundation models},
  author={Li, Kunchang and Wang, Yali and Li, Yizhuo and Wang, Yi and He, Yinan and Wang, Limin and Qiao, Yu},
  journal={arXiv preprint arXiv:2303.16058},
  year={2023}
}
```
