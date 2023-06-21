# STGCN++

[PYSKL: Towards Good Practices for Skeleton Action Recognition](https://arxiv.org/abs/2205.09443)

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

We present PYSKL: an open-source toolbox for skeleton-based action recognition based on PyTorch. The toolbox supports a wide variety of skeleton action recognition algorithms, including approaches based on GCN and CNN. In contrast to existing open-source skeleton action recognition projects that include only one or two algorithms, PYSKL implements six different algorithms under a unified framework with both the latest and original good practices to ease the comparison of efficacy and efficiency. We also provide an original GCN-based skeleton action recognition model named ST-GCN++, which achieves competitive recognition performance without any complicated attention schemes, serving as a strong baseline. Meanwhile, PYSKL supports the training and testing of nine skeleton-based action recognition benchmarks and achieves state-of-the-art recognition performance on eight of them. To facilitate future research on skeleton action recognition, we also provide a large number of trained models and detailed benchmark results to give some insights. PYSKL is released at this https URL and is actively maintained. We will update this report when we add new features or benchmarks. The current version corresponds to PYSKL v0.2.

## Results and Models

### NTU60_XSub_2D

| frame sampling strategy |   modality   | gpus | backbone | top1 acc | testing protocol | FLOPs | params |                  config                   |                  ckpt                   |                  log                   |
| :---------------------: | :----------: | :--: | :------: | :------: | :--------------: | :---: | :----: | :---------------------------------------: | :-------------------------------------: | :------------------------------------: |
|       uniform 100       |    joint     |  8   | STGCN++  |  89.29   |     10 clips     | 1.95G | 1.39M  | [config](/configs/skeleton/stgcnpp/stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcnpp/stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d/stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d_20221228-86e1e77a.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcnpp/stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d/stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.log) |
|       uniform 100       |     bone     |  8   | STGCN++  |  92.30   |     10 clips     | 1.95G | 1.39M  | [config](/configs/skeleton/stgcnpp/stgcnpp_8xb16-bone-u100-80e_ntu60-xsub-keypoint-2d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcnpp/stgcnpp_8xb16-bone-u100-80e_ntu60-xsub-keypoint-2d/stgcnpp_8xb16-bone-u100-80e_ntu60-xsub-keypoint-2d_20221228-cd11a691.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcnpp/stgcnpp_8xb16-bone-u100-80e_ntu60-xsub-keypoint-2d/stgcnpp_8xb16-bone-u100-80e_ntu60-xsub-keypoint-2d.log) |
|       uniform 100       | joint-motion |  8   | STGCN++  |  87.30   |     10 clips     | 1.95G | 1.39M  | [config](/configs/skeleton/stgcnpp/stgcnpp_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-2d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcnpp/stgcnpp_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-2d/stgcnpp_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-2d_20221228-19a34aba.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcnpp/stgcnpp_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-2d/stgcnpp_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-2d.log) |
|       uniform 100       | bone-motion  |  8   | STGCN++  |  88.76   |     10 clips     | 1.95G | 1.39M  | [config](/configs/skeleton/stgcnpp/stgcnpp_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-2d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcnpp/stgcnpp_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-2d/stgcnpp_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-2d_20221228-c02a0749.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcnpp/stgcnpp_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-2d/stgcnpp_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-2d.log) |
|                         |  two-stream  |      |          |  92.61   |                  |       |        |                                           |                                         |                                        |
|                         | four-stream  |      |          |  92.77   |                  |       |        |                                           |                                         |                                        |

### NTU60_XSub_3D

| frame sampling strategy |   modality   | gpus | backbone | top1 acc | testing protocol | FLOPs | params |                  config                   |                  ckpt                   |                  log                   |
| :---------------------: | :----------: | :--: | :------: | :------: | :--------------: | :---: | :----: | :---------------------------------------: | :-------------------------------------: | :------------------------------------: |
|       uniform 100       |    joint     |  8   | STGCN++  |  89.14   |     10 clips     | 2.96G |  1.4M  | [config](/configs/skeleton/stgcnpp/stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-3d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcnpp/stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-3d/stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-3d_20221230-4e455ce3.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcnpp/stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-3d/stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-3d.log) |
|       uniform 100       |     bone     |  8   | STGCN++  |  90.21   |     10 clips     | 2.96G |  1.4M  | [config](/configs/skeleton/stgcnpp/stgcnpp_8xb16-bone-u100-80e_ntu60-xsub-keypoint-3d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcnpp/stgcnpp_8xb16-bone-u100-80e_ntu60-xsub-keypoint-3d/stgcnpp_8xb16-bone-u100-80e_ntu60-xsub-keypoint-3d_20221230-7f356072.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcnpp/stgcnpp_8xb16-bone-u100-80e_ntu60-xsub-keypoint-3d/stgcnpp_8xb16-bone-u100-80e_ntu60-xsub-keypoint-3d.log) |
|       uniform 100       | joint-motion |  8   | STGCN++  |  86.67   |     10 clips     | 2.96G |  1.4M  | [config](/configs/skeleton/stgcnpp/stgcnpp_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-3d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcnpp/stgcnpp_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-3d/stgcnpp_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-3d_20221230-650de5cc.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcnpp/stgcnpp_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-3d/stgcnpp_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-3d.log) |
|       uniform 100       | bone-motion  |  8   | STGCN++  |  87.45   |     10 clips     | 2.96G |  1.4M  | [config](/configs/skeleton/stgcnpp/stgcnpp_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-3d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcnpp/stgcnpp_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-3d/stgcnpp_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-3d_20221230-b00440d2.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcnpp/stgcnpp_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-3d/stgcnpp_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-3d.log) |
|                         |  two-stream  |      |          |  91.39   |                  |       |        |                                           |                                         |                                        |
|                         | four-stream  |      |          |  91.87   |                  |       |        |                                           |                                         |                                        |

1. The **gpus** indicates the number of gpus we used to get the checkpoint. If you want to use a different number of gpus or videos per gpu, the best way is to set `--auto-scale-lr` when calling `tools/train.py`, this parameter will auto-scale the learning rate according to the actual batch size, and the original batch size.
2. For two-stream fusion, we use **joint : bone = 1 : 1**. For four-stream fusion, we use **joint : joint-motion : bone : bone-motion = 2 : 1 : 2 : 1**. For more details about multi-stream fusion, please refer to this [tutorial](/docs/en/useful_tools.md#multi-stream-fusion).

## Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train STGCN++ model on NTU60-2D dataset in a deterministic option with periodic validation.

```shell
python tools/train.py configs/skeleton/stgcnpp/stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py \
    --seed 0 --deterministic
```

For more details, you can refer to the **Training** part in the [Training and Test Tutorial](/docs/en/user_guides/train_test.md).

## Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test STGCN++ model on NTU60-2D dataset and dump the result to a pickle file.

```shell
python tools/test.py configs/skeleton/stgcnpp/stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py \
    checkpoints/SOME_CHECKPOINT.pth --dump result.pkl
```

For more details, you can refer to the **Test** part in the [Training and Test Tutorial](/docs/en/user_guides/train_test.md).

## Citation

```BibTeX
@misc{duan2022PYSKL,
  url = {https://arxiv.org/abs/2205.09443},
  author = {Duan, Haodong and Wang, Jiaqi and Chen, Kai and Lin, Dahua},
  title = {PYSKL: Towards Good Practices for Skeleton Action Recognition},
  publisher = {arXiv},
  year = {2022}
}
```
