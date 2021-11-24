# AGCN

## Introduction

<!-- [ALGORITHM] -->

```BibTeX
@inproceedings{shi2019two,
  title={Two-stream adaptive graph convolutional networks for skeleton-based action recognition},
  author={Shi, Lei and Zhang, Yifan and Cheng, Jian and Lu, Hanqing},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={12026--12035},
  year={2019}
}
```

## Model Zoo

### NTU60_XSub

| config                                                       | type | gpus  |   backbone   | Top-1 |                             ckpt                             |                             log                              |                             json                             |
| :----------------------------------------------------------- | :------------: | :---: | :----------: | :---: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [2sagcn_80e_ntu60_xsub_keypoint_3d](/configs/skeleton/2s-agcn/2sagcn_80e_ntu60_xsub_keypoint_3d.py) |    joint   | 2 | AGCN | 87.31  | [ckpt](https://download.openmmlab.com/mmaction/skeleton/2s-agcn/2sagcn_80e_ntu60_xsub_keypoint_3d/2sagcn_80e_ntu60_xsub_keypoint_3d-e7bb9653.pth) | [log](https://download.openmmlab.com/mmaction/skeleton/2s-agcn/2sagcn_80e_ntu60_xsub_keypoint_3d/2sagcn_80e_ntu60_xsub_keypoint_3d.log) | [json](https://download.openmmlab.com/mmaction/skeleton/2s-agcn/2sagcn_80e_ntu60_xsub_keypoint_3d/2sagcn_80e_ntu60_xsub_keypoint_3d.json) |
| [2sagcn_80e_ntu60_xsub_bone_3d](/configs/skeleton/ss-agcn/2sagcn_80e_ntu60_xsub_bone_3d.py) |    bone    | 2 | AGCN | 87.91  | [ckpt](https://download.openmmlab.com/mmaction/skeleton/2s-agcn/2sagcn_80e_ntu60_xsub_bone_3d/2sagcn_80e_ntu60_xsub_bone_3d-13e7ccf0.pth) | [log](https://download.openmmlab.com/mmaction/skeleton/2s-agcn/2sagcn_80e_ntu60_xsub_bone_3d/2sagcn_80e_ntu60_xsub_bone_3d.log) | [json](https://download.openmmlab.com/mmaction/skeleton/2s-agcn/2sagcn_80e_ntu60_xsub_bone_3d/2sagcn_80e_ntu60_xsub_bone_3d.json) |

## Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train AGCN model on joint data of NTU60 dataset in a deterministic option with periodic validation.

```shell
python tools/train.py configs/skeleton/2s-agcn/2sagcn_80e_ntu60_xsub_keypoint_3d.py \
    --work-dir work_dirs/2sagcn_80e_ntu60_xsub_keypoint_3d \
    --validate --seed 0 --deterministic
```

Example: train AGCN model on bone data of NTU60 dataset in a deterministic option with periodic validation.

```shell
python tools/train.py configs/skeleton/2s-agcn/2sagcn_80e_ntu60_xsub_bone_3d.py \
    --work-dir work_dirs/2sagcn_80e_ntu60_xsub_bone_3d \
    --validate --seed 0 --deterministic
```

For more details, you can refer to **Training setting** part in [getting_started](/docs/getting_started.md#training-setting).

## Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test AGCN model on joint data of NTU60 dataset and dump the result to a pickle file.

```shell
python tools/test.py configs/skeleton/2s-agcn/2sagcn_80e_ntu60_xsub_keypoint_3d.py \
    checkpoints/SOME_CHECKPOINT.pth --eval top_k_accuracy mean_class_accuracy \
    --out joint_result.pkl
```

Example: test AGCN model on bone data of NTU60 dataset and dump the result to a pickle file.

```shell
python tools/test.py configs/skeleton/2s-agcn/2sagcn_80e_ntu60_xsub_bone_3d.py \
    checkpoints/SOME_CHECKPOINT.pth --eval top_k_accuracy mean_class_accuracy \
    --out bone_result.pkl
```

For more details, you can refer to **Test a dataset** part in [getting_started](/docs/getting_started.md#test-a-dataset).
