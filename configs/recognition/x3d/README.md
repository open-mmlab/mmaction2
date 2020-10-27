# X3d

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
|[x3d_s_13x6x1_facebook_kinetics400_rgb](/configs/recognition/x3d/x3d_s_13x6x1_facebook_kinetics400_rgb.py)|short-side 320| X3D_S | 72.7 | 73.2 | [73.1](https://github.com/facebookresearch/SlowFast/blob/master/MODEL_ZOO.md) | [73.5](https://github.com/facebookresearch/SlowFast/blob/master/MODEL_ZOO.md) | [ckpt]()[1] |
|[x3d_m_16x5x1_facebook_kinetics400_rgb](/configs/recognition/x3d/x3d_m_16x5x1_facebook_kinetics400_rgb)|short-side 320| X3D_M | 75.0 | 75.6 | [75.1](https://github.com/facebookresearch/SlowFast/blob/master/MODEL_ZOO.md) | [76.2](https://github.com/facebookresearch/SlowFast/blob/master/MODEL_ZOO.md) | [ckpt]()[1] |

[1] The models are ported from the repo [SlowFast](https://github.com/facebookresearch/SlowFast/) and tested on our data. Currently, we only support the testing of X3d models, training will be available soon.

Notes:

1. The **gpus** indicates the number of gpu we used to get the checkpoint. It is noteworthy that the configs we provide are used for 8 gpus as default.
According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you may set the learning rate proportional to the batch size if you use different GPUs or videos per GPU,
e.g., lr=0.01 for 4 GPUs * 2 video/gpu and lr=0.08 for 16 GPUs * 4 video/gpu.
2. The **inference_time** is got by this [benchmark script](/tools/analysis/benchmark.py), where we use the sampling frames strategy of the test setting and only care about the model inference time,
not including the IO time and pre-processing time. For each setting, we use 1 gpu and set batch size (videos per gpu) to 1 to calculate the inference time.
3. The values in columns named after "reference" are the results got by testing the checkpoint released on the original repo and codes, using the same dataset with ours.

For more details on data preparation, you can refer to Kinetics400 in [Data Preparation](/docs/data_preparation.md).

## Test
You can use the following command to test a model.
```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test X3d model on Kinetics-400 dataset and dump the result to a json file.
```shell
python tools/test.py configs/recognition/x3d/x3d_s_13x6x1_facebook_kinetics400_rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --eval top_k_accuracy mean_class_accuracy \
    --out result.json --average-clips prob
```

For more details, you can refer to **Test a dataset** part in [getting_started](/docs/getting_started.md#test-a-dataset).
