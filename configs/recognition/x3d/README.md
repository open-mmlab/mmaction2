# X3D

## Introduction
```
@misc{feichtenhofer2020x3d,
      title={X3D: Expanding Architectures for Efficient Video Recognition},
      author={Christoph Feichtenhofer},
      year={2020},
      eprint={2004.04730},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Model Zoo

### Kinetics-400

|config | resolution | backbone | top1 10-view | top1 30-view | reference top1 10-view | reference top1 30-view | ckpt |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|[x3d_s_13x6x1_facebook_kinetics400_rgb](/configs/recognition/x3d/x3d_s_13x6x1_facebook_kinetics400_rgb.py)|short-side 320| X3D_S | 72.7 | 73.2 | 73.1 [[SlowFast](https://github.com/facebookresearch/SlowFast/blob/master/MODEL_ZOO.md)] | 73.5 [[SlowFast](https://github.com/facebookresearch/SlowFast/blob/master/MODEL_ZOO.md)] | [ckpt](https://download.openmmlab.com/mmaction/recognition/x3d/facebook/x3d_s_facebook_13x6x1_kinetics400_rgb_20201027-623825a0.pth)[1] |
|[x3d_m_16x5x1_facebook_kinetics400_rgb](/configs/recognition/x3d/x3d_m_16x5x1_facebook_kinetics400_rgb)|short-side 320| X3D_M | 75.0 | 75.6 | 75.1 [[SlowFast](https://github.com/facebookresearch/SlowFast/blob/master/MODEL_ZOO.md)] | 76.2 [[SlowFast](https://github.com/facebookresearch/SlowFast/blob/master/MODEL_ZOO.md)] | [ckpt](https://download.openmmlab.com/mmaction/recognition/x3d/facebook/x3d_m_facebook_16x5x1_kinetics400_rgb_20201027-3f42382a.pth)[1] |

[1] The models are ported from the repo [SlowFast](https://github.com/facebookresearch/SlowFast/) and tested on our data. Currently, we only support the testing of X3D models, training will be available soon.

Notes:

3. The values in columns named after "reference" are the results got by testing the checkpoint released on the original repo and codes, using the same dataset with ours.

For more details on data preparation, you can refer to Kinetics400 in [Data Preparation](/docs/data_preparation.md).

## Test
You can use the following command to test a model.
```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test X3D model on Kinetics-400 dataset and dump the result to a json file.
```shell
python tools/test.py configs/recognition/x3d/x3d_s_13x6x1_facebook_kinetics400_rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --eval top_k_accuracy mean_class_accuracy \
    --out result.json --average-clips prob
```

For more details, you can refer to **Test a dataset** part in [getting_started](/docs/getting_started.md#test-a-dataset).
