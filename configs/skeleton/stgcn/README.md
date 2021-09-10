# STGCN

## Introduction

<!-- [ALGORITHM] -->

```BibTeX
@inproceedings{yan2018spatial,
  title={Spatial temporal graph convolutional networks for skeleton-based action recognition},
  author={Yan, Sijie and Xiong, Yuanjun and Lin, Dahua},
  booktitle={Thirty-second AAAI conference on artificial intelligence},
  year={2018}
}
```

## Model Zoo

### NTU60_XSub

| config                                                       | pseudo heatmap | gpus  |   backbone   | Top-1 |                             ckpt                             |                             log                              |                             json                             |
| :----------------------------------------------------------- | :------------: | :---: | :----------: | :---: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [stgcn_ntu_rgbd60_xsub](/configs/skeleton/stgcn/stgcn_ntu_rgbd60_xsub.py) |    keypoint    | 2 | STGCN | 86.91  | [ckpt](https://download.openmmlab.com/mmaction/skeleton/stgcn/stgcn_ntu_rgbd60_xsub/stgcn_ntu_rgbd60_xsub-f3adabf1.pth) | [log](https://download.openmmlab.com/mmaction/skeleton/stgcn/stgcn_ntu_rgbd60_xsub/stgcn_ntu_rgbd60_xsub.log) | [json](https://download.openmmlab.com/mmaction/skeleton/stgcn/stgcn_ntu_rgbd60_xsub/stgcn_ntu_rgbd60_xsub.json) |

## Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train STGCN model on NTU60 dataset in a deterministic option with periodic validation.

```shell
python tools/train.py configs/skeleton/stgcn/stgcn_ntu_rgbd60_xsub.py \
    --work-dir work_dirs/stgcn_ntu_rgbd60_xsub \
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
python tools/test.py configs/skeleton/stgcn/stgcn_ntu_rgbd60_xsub.py \
    checkpoints/SOME_CHECKPOINT.pth --eval top_k_accuracy mean_class_accuracy \
    --out result.pkl
```

For more details, you can refer to **Test a dataset** part in [getting_started](/docs/getting_started.md#test-a-dataset).
