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

### NTU60_XSub

| frame sampling strategy | keypoint | gpus | backbone | top1 acc |                        config                        |                        ckpt                        |                        log                         |
| :---------------------: | :-:|:------: | :--: | :------: | :---: | :--------------------------------------------------: | :------------------------------------------------: | :------------------------------------------------: |
|       Uniform-100       |    Joint-3d    |  8   |  STGCN   | 86.91 | [config](/configs/skeleton/stgcn/stgcn_1xb16-80e_ntu60-xsub-keypoint.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_1xb16-80e_ntu60-xsub-keypoint/stgcn_1xb16-80e_ntu60-xsub-keypoint-e7bb9653.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_1xb16-80e_ntu60-xsub-keypoint/stgcn_1xb16-80e_ntu60-xsub-keypoint.log) |
|       Uniform-100       |    Bone-3d    |  8   |  STGCN   | 86.91 | [config](/configs/skeleton/stgcn/stgcn_1xb16-80e_ntu60-xsub-keypoint.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_1xb16-80e_ntu60-xsub-keypoint/stgcn_1xb16-80e_ntu60-xsub-keypoint-e7bb9653.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_1xb16-80e_ntu60-xsub-keypoint/stgcn_1xb16-80e_ntu60-xsub-keypoint.log) |
|       Uniform-100       |    Joint-Motion-3d    |  8   |  STGCN   | 86.91 | [config](/configs/skeleton/stgcn/stgcn_1xb16-80e_ntu60-xsub-keypoint.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_1xb16-80e_ntu60-xsub-keypoint/stgcn_1xb16-80e_ntu60-xsub-keypoint-e7bb9653.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_1xb16-80e_ntu60-xsub-keypoint/stgcn_1xb16-80e_ntu60-xsub-keypoint.log) |
|       Uniform-100       |    Bone-Motion-3d    |  8   |  STGCN   | 86.91 | [config](/configs/skeleton/stgcn/stgcn_1xb16-80e_ntu60-xsub-keypoint.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_1xb16-80e_ntu60-xsub-keypoint/stgcn_1xb16-80e_ntu60-xsub-keypoint-e7bb9653.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_1xb16-80e_ntu60-xsub-keypoint/stgcn_1xb16-80e_ntu60-xsub-keypoint.log) |

|       Zero-Pad-300       |    3d    |  8   |  STGCN   | 86.91 | [config](/configs/skeleton/stgcn/stgcn_1xb32-80e-ntu60_xsub-keypoint-3d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_1xb32-80e-ntu60_xsub-keypoint-3d/stgcn_1xb32-80e-ntu60_xsub-keypoint-3d-13e7ccf0.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_1xb32-80e-ntu60_xsub-keypoint-3d/stgcn_1xb32-80e-ntu60_xsub-keypoint-3d.log) |

\* The number is copied from the [paper](https://arxiv.org/pdf/2106.09696.pdf), the performance of the [released checkpoints](https://github.com/abhinanda-punnakkal/BABEL/tree/main/action_recognition) for BABEL-120 is inferior.

## Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train STGCN model on NTU60 dataset in a deterministic option with periodic validation.

```shell
python tools/train.py configs/skeleton/stgcn/stgcn_1xb16-80e_ntu60-xsub-keypoint.py \
    --work-dir work_dirs/stgcn_1xb16-80e_ntu60-xsub-keypoint \
    --cfg-options randomness.seed=0 randomness.deterministic=True
```

For more details, you can refer to the **Training** part in the [Training and Test Tutorial](/docs/en/user_guides/4_train_test.md).

## Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test STGCN model on NTU60 dataset and dump the result to a pickle file.

```shell
python tools/test.py configs/skeleton/stgcn/stgcn_1xb16-80e_ntu60-xsub-keypoint.py \
    checkpoints/SOME_CHECKPOINT.pth --eval top_k_accuracy mean_class_accuracy \
    --out result.pkl
```

For more details, you can refer to the **Test** part in the [Training and Test Tutorial](/docs/en/user_guides/4_train_test.md).

## Citation

```BibTeX
@inproceedings{yan2018spatial,
  title={Spatial temporal graph convolutional networks for skeleton-based action recognition},
  author={Yan, Sijie and Xiong, Yuanjun and Lin, Dahua},
  booktitle={Thirty-second AAAI conference on artificial intelligence},
  year={2018}
}
```
