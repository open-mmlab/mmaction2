# STGCN

[Spatial temporal graph convolutional networks for skeleton-based action recognition](https://ojs.aaai.org/index.php/AAAI/article/view/12328)

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

Dynamics of human body skeletons convey significant information for human action recognition. Conventional approaches for modeling skeletons usually rely on hand-crafted parts or traversal rules, thus resulting in limited expressive power and difficulties of generalization. In this work, we propose a novel model of dynamic skeletons called Spatial-Temporal Graph Convolutional Networks (ST-GCN), which moves beyond the limitations of previous methods by automatically learning both the spatial and temporal patterns from data. This formulation not only leads to greater expressive power but also stronger generalization capability. On two large datasets, Kinetics and NTU-RGBD, it achieves substantial improvements over mainstream methods.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/34324155/142995893-d6618728-072c-46e1-b276-9b88cf21a01c.png" width="800"/>
</div>

## Results and Models

### NTU60_XSub_2D

| frame sampling strategy |   modality   | gpus | backbone | top1 acc | testing protocol | FLOPs | params |                  config                   |                  ckpt                   |                  log                   |
| :---------------------: | :----------: | :--: | :------: | :------: | :--------------: | :---: | :----: | :---------------------------------------: | :-------------------------------------: | :------------------------------------: |
|       uniform 100       |    joint     |  8   |  STGCN   |  88.95   |     10 clips     | 3.8G  |  3.1M  | [config](/configs/skeleton/stgcn/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d_20221129-484a394a.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.log) |
|       uniform 100       |     bone     |  8   |  STGCN   |  91.69   |     10 clips     | 3.8G  |  3.1M  | [config](/configs/skeleton/stgcn/stgcn_8xb16-bone-u100-80e_ntu60-xsub-keypoint-2d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-bone-u100-80e_ntu60-xsub-keypoint-2d/stgcn_8xb16-bone-u100-80e_ntu60-xsub-keypoint-2d_20221129-c4b44488.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-bone-u100-80e_ntu60-xsub-keypoint-2d/stgcn_8xb16-bone-u100-80e_ntu60-xsub-keypoint-2d.log) |
|       uniform 100       | joint-motion |  8   |  STGCN   |  86.90   |     10 clips     | 3.8G  |  3.1M  | [config](/configs/skeleton/stgcn/stgcn_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-2d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-2d/stgcn_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-2d_20221129-f18eb408.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-2d/stgcn_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-2d.log) |
|       uniform 100       | bone-motion  |  8   |  STGCN   |  87.86   |     10 clips     | 3.8G  |  3.1M  | [config](/configs/skeleton/stgcn/stgcn_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-2d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-2d/stgcn_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-2d_20221129-99c60e2d.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-2d/stgcn_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-2d.log) |
|                         |  two-stream  |      |          |  92.12   |                  |       |        |                                           |                                         |                                        |
|                         | four-stream  |      |          |  92.34   |                  |       |        |                                           |                                         |                                        |

### NTU60_XSub_3D

| frame sampling strategy |   modality   | gpus | backbone | top1 acc | testing protocol | FLOPs | params |                  config                   |                  ckpt                   |                  log                   |
| :---------------------: | :----------: | :--: | :------: | :------: | :--------------: | :---: | :----: | :---------------------------------------: | :-------------------------------------: | :------------------------------------: |
|       uniform 100       |    joint     |  8   |  STGCN   |  88.11   |     10 clips     | 5.7G  |  3.1M  | [config](/configs/skeleton/stgcn/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-3d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-3d/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-3d_20221129-850308e1.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-3d/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-3d.log) |
|       uniform 100       |     bone     |  8   |  STGCN   |  88.76   |     10 clips     | 5.7G  |  3.1M  | [config](/configs/skeleton/stgcn/stgcn_8xb16-bone-u100-80e_ntu60-xsub-keypoint-3d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-bone-u100-80e_ntu60-xsub-keypoint-3d/stgcn_8xb16-bone-u100-80e_ntu60-xsub-keypoint-3d_20221129-9c8d2970.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-bone-u100-80e_ntu60-xsub-keypoint-3d/stgcn_8xb16-bone-u100-80e_ntu60-xsub-keypoint-3d.log) |
|       uniform 100       | joint-motion |  8   |  STGCN   |  86.06   |     10 clips     | 5.7G  |  3.1M  | [config](/configs/skeleton/stgcn/stgcn_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-3d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-3d/stgcn_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-3d_20221129-927648ea.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-3d/stgcn_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-3d.log) |
|       uniform 100       | bone-motion  |  8   |  STGCN   |  85.49   |     10 clips     | 5.7G  |  3.1M  | [config](/configs/skeleton/stgcn/stgcn_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-3d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-3d/stgcn_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-3d_20221129-593162ca.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-3d/stgcn_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-3d.log) |
|                         |  two-stream  |      |          |  90.14   |                  |       |        |                                           |                                         |                                        |
|                         | four-stream  |      |          |  90.39   |                  |       |        |                                           |                                         |                                        |

### NTU120_XSub_2D

| frame sampling strategy |   modality   | gpus | backbone | top1 acc | testing protocol | FLOPs | params |                  config                   |                  ckpt                   |                  log                   |
| :---------------------: | :----------: | :--: | :------: | :------: | :--------------: | :---: | :----: | :---------------------------------------: | :-------------------------------------: | :------------------------------------: |
|       uniform 100       |    joint     |  8   |  STGCN   |  83.19   |     10 clips     | 3.8G  |  3.1M  | [config](/configs/skeleton/stgcn/stgcn_8xb16-joint-u100-80e_ntu120-xsub-keypoint-2d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-joint-u100-80e_ntu120-xsub-keypoint-2d/stgcn_8xb16-joint-u100-80e_ntu120-xsub-keypoint-2d_20221129-612416c6.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-joint-u100-80e_ntu120-xsub-keypoint-2d/stgcn_8xb16-joint-u100-80e_ntu120-xsub-keypoint-2d.log) |
|       uniform 100       |     bone     |  8   |  STGCN   |  83.36   |     10 clips     | 3.8G  |  3.1M  | [config](/configs/skeleton/stgcn/stgcn_8xb16-bone-u100-80e_ntu120-xsub-keypoint-2d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-bone-u100-80e_ntu120-xsub-keypoint-2d/stgcn_8xb16-bone-u100-80e_ntu120-xsub-keypoint-2d_20221129-131e63c3.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-bone-u100-80e_ntu120-xsub-keypoint-2d/stgcn_8xb16-bone-u100-80e_ntu120-xsub-keypoint-2d.log) |
|       uniform 100       | joint-motion |  8   |  STGCN   |  78.87   |     10 clips     | 3.8G  |  3.1M  | [config](/configs/skeleton/stgcn/stgcn_8xb16-joint-motion-u100-80e_ntu120-xsub-keypoint-2d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-joint-motion-u100-80e_ntu120-xsub-keypoint-2d/stgcn_8xb16-joint-motion-u100-80e_ntu120-xsub-keypoint-2d_20221129-7cb38ec2.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-joint-motion-u100-80e_ntu120-xsub-keypoint-2d/stgcn_8xb16-joint-motion-u100-80e_ntu120-xsub-keypoint-2d.log) |
|       uniform 100       | bone-motion  |  8   |  STGCN   |  79.55   |     10 clips     | 3.8G  |  3.1M  | [config](/configs/skeleton/stgcn/stgcn_8xb16-bone-motion-u100-80e_ntu120-xsub-keypoint-2d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-bone-motion-u100-80e_ntu120-xsub-keypoint-2d/stgcn_8xb16-bone-motion-u100-80e_ntu120-xsub-keypoint-2d_20221129-f5b19892.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-bone-motion-u100-80e_ntu120-xsub-keypoint-2d/stgcn_8xb16-bone-motion-u100-80e_ntu120-xsub-keypoint-2d.log) |
|                         |  two-stream  |      |          |  84.84   |                  |       |        |                                           |                                         |                                        |
|                         | four-stream  |      |          |  85.23   |                  |       |        |                                           |                                         |                                        |

### NTU120_XSub_3D

| frame sampling strategy |   modality   | gpus | backbone | top1 acc | testing protocol | FLOPs | params |                  config                   |                  ckpt                   |                  log                   |
| :---------------------: | :----------: | :--: | :------: | :------: | :--------------: | :---: | :----: | :---------------------------------------: | :-------------------------------------: | :------------------------------------: |
|       uniform 100       |    joint     |  8   |  STGCN   |  82.15   |     10 clips     | 5.7G  |  3.1M  | [config](/configs/skeleton/stgcn/stgcn_8xb16-joint-u100-80e_ntu120-xsub-keypoint-3d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-joint-u100-80e_ntu120-xsub-keypoint-3d/stgcn_8xb16-joint-u100-80e_ntu120-xsub-keypoint-3d_20221129-0484f579.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-joint-u100-80e_ntu120-xsub-keypoint-3d/stgcn_8xb16-joint-u100-80e_ntu120-xsub-keypoint-3d.log) |
|       uniform 100       |     bone     |  8   |  STGCN   |  84.28   |     10 clips     | 5.7G  |  3.1M  | [config](/configs/skeleton/stgcn/stgcn_8xb16-bone-u100-80e_ntu120-xsub-keypoint-3d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-bone-u100-80e_ntu120-xsub-keypoint-3d/stgcn_8xb16-bone-u100-80e_ntu120-xsub-keypoint-3d_20221129-bc007510.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-bone-u100-80e_ntu120-xsub-keypoint-3d/stgcn_8xb16-bone-u100-80e_ntu120-xsub-keypoint-3d.log) |
|       uniform 100       | joint-motion |  8   |  STGCN   |  78.93   |     10 clips     | 5.7G  |  3.1M  | [config](/configs/skeleton/stgcn/stgcn_8xb16-joint-motion-u100-80e_ntu120-xsub-keypoint-3d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-joint-motion-u100-80e_ntu120-xsub-keypoint-3d/stgcn_8xb16-joint-motion-u100-80e_ntu120-xsub-keypoint-3d_20221129-5d54f525.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-joint-motion-u100-80e_ntu120-xsub-keypoint-3d/stgcn_8xb16-joint-motion-u100-80e_ntu120-xsub-keypoint-3d.log) |
|       uniform 100       | bone-motion  |  8   |  STGCN   |  80.02   |     10 clips     | 5.7G  |  3.1M  | [config](/configs/skeleton/stgcn/stgcn_8xb16-bone-motion-u100-80e_ntu120-xsub-keypoint-3d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-bone-motion-u100-80e_ntu120-xsub-keypoint-3d/stgcn_8xb16-bone-motion-u100-80e_ntu120-xsub-keypoint-3d_20221129-3cb0e4e1.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-bone-motion-u100-80e_ntu120-xsub-keypoint-3d/stgcn_8xb16-bone-motion-u100-80e_ntu120-xsub-keypoint-3d.log) |
|                         |  two-stream  |      |          |  85.68   |                  |       |        |                                           |                                         |                                        |
|                         | four-stream  |      |          |  86.19   |                  |       |        |                                           |                                         |                                        |

1. The **gpus** indicates the number of gpus we used to get the checkpoint. If you want to use a different number of gpus or videos per gpu, the best way is to set `--auto-scale-lr` when calling `tools/train.py`, this parameter will auto-scale the learning rate according to the actual batch size, and the original batch size.
2. For two-stream fusion, we use **joint : bone = 1 : 1**. For four-stream fusion, we use **joint : joint-motion : bone : bone-motion = 2 : 1 : 2 : 1**. For more details about multi-stream fusion, please refer to this [tutorial](/docs/en/useful_tools.md#multi-stream-fusion.md).

## Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train STGCN model on NTU60-2D dataset in a deterministic option with periodic validation.

```shell
python tools/train.py configs/skeleton/stgcn/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py \
    --seed 0 --deterministic
```

For more details, you can refer to the **Training** part in the [Training and Test Tutorial](/docs/en/user_guides/train_test.md).

## Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test STGCN model on NTU60-2D dataset and dump the result to a pickle file.

```shell
python tools/test.py configs/skeleton/stgcn/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py \
    checkpoints/SOME_CHECKPOINT.pth --dump result.pkl
```

For more details, you can refer to the **Test** part in the [Training and Test Tutorial](/docs/en/user_guides/train_test.md).

## Citation

```BibTeX
@inproceedings{yan2018spatial,
  title={Spatial temporal graph convolutional networks for skeleton-based action recognition},
  author={Yan, Sijie and Xiong, Yuanjun and Lin, Dahua},
  booktitle={Thirty-second AAAI conference on artificial intelligence},
  year={2018}
}
```
