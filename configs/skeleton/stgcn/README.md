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

| frame sampling strategy | keypoint | gpus | backbone | Top-1 |                        config                        |                        ckpt                        |                        log                         |
| :---------------------: | :------: | :--: | :------: | :---: | :--------------------------------------------------: | :------------------------------------------------: | :------------------------------------------------: |
|       padding 300       |    2d    |  1   |  STGCN   | 86.91 | [config](/configs/skeleton/stgcn/stgcn_1xb16-80e_ntu60-xsub-keypoint.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_1xb16-80e_ntu60-xsub-keypoint/stgcn_1xb16-80e_ntu60-xsub-keypoint-e7bb9653.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_1xb16-80e_ntu60-xsub-keypoint/stgcn_1xb16-80e_ntu60-xsub-keypoint.log) |
|       padding 300       |    3d    |  1   |  STGCN   | 86.91 | [config](/configs/skeleton/stgcn/stgcn_1xb32-80e-ntu60_xsub-keypoint-3d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_1xb32-80e-ntu60_xsub-keypoint-3d/stgcn_1xb32-80e-ntu60_xsub-keypoint-3d-13e7ccf0.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_1xb32-80e-ntu60_xsub-keypoint-3d/stgcn_1xb32-80e-ntu60_xsub-keypoint-3d.log) |

### BABEL

|   dataset    | gpus | backbone |   Top-1   | Mean Top-1 | Top-1 Official (AGCN) | Mean Top-1 Official (AGCN) |                 config                 |                 ckpt                 |                 log                  |
| :----------: | :--: | :------: | :-------: | :--------: | :-------------------: | :------------------------: | :------------------------------------: | :----------------------------------: | :----------------------------------: |
|   babel60    |  8   |  STGCN   | **42.39** | **28.28**  |         41.14         |           24.46            | [config](/configs/skeleton/stgcn/stgcn_8xb16-80e_babel60.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-80e_babel60/stgcn_8xb16-80e_babel60-3d206418.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-80e_babel60/stgcn_8xb16-80e_babel60.log) |
| babel60-wfl  |  8   |  STGCN   | **40.31** |   29.79    |         33.41         |         **30.42**          | [config](/configs/skeleton/stgcn/stgcn_8xb16-80e-babel60-wfl.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-80e-babel60-wfl/stgcn_8xb16-80e-babel60-wfl-1a9102d7.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-80e_babel60/stgcn_8xb16-80e-babel60-wfl.log) |
|   babel120   |  8   |  ST-GCN  | **38.95** | **20.58**  |         38.41         |           17.56            | [config](/configs/skeleton/stgcn/stgcn_8xb16-80e_babel120.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-80e_babel120/stgcn_8xb16-80e_babel120-e41eb6d7.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-80e_babel60/stgcn_8xb16-80e_babel120.log) |
| babel120-wfl |  8   |  ST-GCN  | **33.00** |   24.33    |         27.91         |        **26.17**\*         | [config](/configs/skeleton/stgcn/stgcn_8xb16-80e_babel120-wfl.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-80e_babel120-wfl/stgcn_8xb16-80e_babel120-wfl-3f2c100d.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-80e_babel60/stgcn_8xb16-80e_babel120-wfl.log) |

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
