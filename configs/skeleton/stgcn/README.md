# STGCN

[Spatial temporal graph convolutional networks for skeleton-based action recognition](https://jivp-eurasipjournals.springeropen.com/articles/10.1186/s13640-019-0476-x)

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

| config                                                       | keypoint | gpus  |   backbone   | Top-1 |                             ckpt                             |                             log                              |                             json                             |
| :----------------------------------------------------------- | :------------: | :---: | :----------: | :---: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [stgcn_80e_ntu60_xsub_keypoint](/configs/skeleton/stgcn/stgcn_80e_ntu60_xsub_keypoint.py) |    2d   | 2 | STGCN | 86.91  | [ckpt](https://download.openmmlab.com/mmaction/skeleton/stgcn/stgcn_80e_ntu60_xsub_keypoint/stgcn_80e_ntu60_xsub_keypoint-e7bb9653.pth) | [log](https://download.openmmlab.com/mmaction/skeleton/stgcn/stgcn_80e_ntu60_xsub_keypoint/stgcn_80e_ntu60_xsub_keypoint.log) | [json](https://download.openmmlab.com/mmaction/skeleton/stgcn/stgcn_80e_ntu60_xsub_keypoint/stgcn_80e_ntu60_xsub_keypoint.json) |
| [stgcn_80e_ntu60_xsub_keypoint_3d](/configs/skeleton/stgcn/stgcn_80e_ntu60_xsub_keypoint_3d.py) |    3d    | 1 | STGCN | 84.61  | [ckpt](https://download.openmmlab.com/mmaction/skeleton/stgcn/stgcn_80e_ntu60_xsub_keypoint_3d/stgcn_80e_ntu60_xsub_keypoint_3d-13e7ccf0.pth) | [log](https://download.openmmlab.com/mmaction/skeleton/stgcn/stgcn_80e_ntu60_xsub_keypoint_3d/stgcn_80e_ntu60_xsub_keypoint_3d.log) | [json](https://download.openmmlab.com/mmaction/skeleton/stgcn/stgcn_80e_ntu60_xsub_keypoint_3d/stgcn_80e_ntu60_xsub_keypoint_3d.json) |

### BABEL

| config                                                       | gpus | backbone |   Top-1   | Mean Top-1 | Top-1 Official (AGCN) | Mean Top-1 Official (AGCN) |                             ckpt                             |                             log                              |
| ------------------------------------------------------------ | :--: | :------: | :-------: | :--------: | :-------------------: | :------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [stgcn_80e_babel60](/configs/skeleton/stgcn/stgcn_80e_babel60.py) |  8   |  ST-GCN  | **42.39** | **28.28**  |         41.14         |           24.46            | [ckpt](https://download.openmmlab.com/mmaction/skeleton/stgcn/stgcn_80e_babel60/stgcn_80e_babel60-3d206418.pth) | [log](https://download.openmmlab.com/mmaction/skeleton/stgcn/stgcn_80e_babel60/stgcn_80e_babel60.log) |
| [stgcn_80e_babel60_wfl](/configs/skeleton/stgcn/stgcn_80e_babel60_wfl.py) |  8   |  ST-GCN  | **40.31** |   29.79    |         33.41         |         **30.42**          | [ckpt](https://download.openmmlab.com/mmaction/skeleton/stgcn/stgcn_80e_babel60_wfl/stgcn_80e_babel60_wfl-1a9102d7.pth) | [log](https://download.openmmlab.com/mmaction/skeleton/stgcn/stgcn_80e_babel60/stgcn_80e_babel60_wfl.log) |
| [stgcn_80e_babel120](/configs/skeleton/stgcn/stgcn_80e_babel120.py) |  8   |  ST-GCN  | **38.95** | **20.58**  |         38.41         |           17.56            | [ckpt](https://download.openmmlab.com/mmaction/skeleton/stgcn/stgcn_80e_babel120/stgcn_80e_babel120-e41eb6d7.pth) | [log](https://download.openmmlab.com/mmaction/skeleton/stgcn/stgcn_80e_babel60/stgcn_80e_babel120.log) |
| [stgcn_80e_babel120_wfl](/configs/skeleton/stgcn/stgcn_80e_babel120_wfl.py) |  8   |  ST-GCN  | **33.00** |   24.33    |         27.91         |         **26.17***         | [ckpt](https://download.openmmlab.com/mmaction/skeleton/stgcn/stgcn_80e_babel120_wfl/stgcn_80e_babel120_wfl-3f2c100d.pth) | [log](https://download.openmmlab.com/mmaction/skeleton/stgcn/stgcn_80e_babel60/stgcn_80e_babel120_wfl.log) |

\* The number is copied from the [paper](https://arxiv.org/pdf/2106.09696.pdf), the performance of the [released checkpoints](https://github.com/abhinanda-punnakkal/BABEL/tree/main/action_recognition) for BABEL-120 is inferior.

## Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train STGCN model on NTU60 dataset in a deterministic option with periodic validation.

```shell
python tools/train.py configs/skeleton/stgcn/stgcn_80e_ntu60_xsub_keypoint.py \
    --work-dir work_dirs/stgcn_80e_ntu60_xsub_keypoint \
    --validate --seed 0 --deterministic
```

For more details, you can refer to **Training setting** part in [getting_started](/docs/getting_started.md#training-setting).

## Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test STGCN model on NTU60 dataset and dump the result to a pickle file.

```shell
python tools/test.py configs/skeleton/stgcn/stgcn_80e_ntu60_xsub_keypoint.py \
    checkpoints/SOME_CHECKPOINT.pth --eval top_k_accuracy mean_class_accuracy \
    --out result.pkl
```

For more details, you can refer to **Test a dataset** part in [getting_started](/docs/getting_started.md#test-a-dataset).

## Citation

```BibTeX
@inproceedings{yan2018spatial,
  title={Spatial temporal graph convolutional networks for skeleton-based action recognition},
  author={Yan, Sijie and Xiong, Yuanjun and Lin, Dahua},
  booktitle={Thirty-second AAAI conference on artificial intelligence},
  year={2018}
}
```
