# Getting Started

This page provides basic tutorials about the usage of MMAction2.
For installation instructions, please see [install.md](install.md).

<!-- TOC -->

- [Datasets](#datasets)
- [Inference with Pre-Trained Models](#inference-with-pre-trained-models)
  * [Test a dataset](#test-a-dataset)
  * [High-level APIs for testing a video and rawframes.](#high-level-apis-for-testing-a-video-and-rawframes)
- [Build a Model](#build-a-model)
  * [Build a model with basic components](#build-a-model-with-basic-components)
  * [Write a new model](#write-a-new-model)
- [Train a Model](#train-a-model)
  * [Iteration pipeline](#iteration-pipeline)
  * [Training setting](#training-setting)
  * [Train with a single GPU](#train-with-a-single-gpu)
  * [Train with multiple GPUs](#train-with-multiple-gpus)
  * [Train with multiple machines](#train-with-multiple-machines)
  * [Launch multiple jobs on a single machine](#launch-multiple-jobs-on-a-single-machine)
- [Tutorials](#tutorials)

<!-- TOC -->

## Datasets

It is recommended to symlink the dataset root to `$MMACTION2/data`.
If your folder structure is different, you may need to change the corresponding paths in config files.

```
mmaction2
├── mmaction
├── tools
├── configs
├── data
│   ├── kinetics400
│   │   ├── rawframes_train
│   │   ├── rawframes_val
│   │   ├── kinetics_train_list.txt
│   │   ├── kinetics_val_list.txt
│   ├── ucf101
│   │   ├── rawframes_train
│   │   ├── rawframes_val
│   │   ├── ucf101_train_list.txt
│   │   ├── ucf101_val_list.txt
│   ├── ...
```
For more information on data preparation, please see [data_preparation.md](data_preparation.md)

For using custom datasets, please refer to [Tutorial 2: Adding New Dataset](tutorials/new_dataset.md)

## Inference with Pre-Trained Models

We provide testing scripts to evaluate a whole dataset (Kinetics-400, Something-Something V1&V2, (Multi-)Moments in Time, etc.),
and provide some high-level apis for easier integration to other projects.

### Test a dataset

- [x] single GPU
- [x] single node multiple GPUs
- [x] multiple node

You can use the following commands to test a dataset.

```shell
# single-gpu testing
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}] \
    [--gpu-collect] [--tmpdir ${TMPDIR}] [--options ${OPTIONS}] [--average-clips ${AVG_TYPE}] \
    [--launcher ${JOB_LAUNCHER}] [--local_rank ${LOCAL_RANK}]

# multi-gpu testing
./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}] \
    [--gpu-collect] [--tmpdir ${TMPDIR}] [--options ${OPTIONS}] [--average-clips ${AVG_TYPE}] \
    [--launcher ${JOB_LAUNCHER}] [--local_rank ${LOCAL_RANK}]
```

Optional arguments:
- `RESULT_FILE`: Filename of the output results. If not specified, the results will not be saved to a file.
- `EVAL_METRICS`: Items to be evaluated on the results. Allowed values depend on the dataset, e.g., `top_k_accuracy`, `mean_class_accuracy` are available for all datasets in recognition, `mmit_mean_average_precision` for Multi-Moments in Time, `mean_average_precision` for Multi-Moments in Time and HVU single category. `AR@AN` for ActivityNet, etc.
- `--gpu-collect`: If specified, recognition results will be collected using gpu communication. Otherwise, it will save the results on different gpus to `TMPDIR` and collect them by the rank 0 worker.
- `TMPDIR`: Temporary directory used for collecting results from multiple workers, available when `--gpu-collect` is not specified.
- `OPTIONS`: Custom options used for evaluation. Allowed values depend on the arguments of the `evaluate` function in dataset.
- `AVG_TYPE`: Items to average the test clips. If set to `prob`, it will apply softmax before averaging the clip scores. Otherwise, it will directly average the clip scores.
- `JOB_LAUNCHER`: Items for distributed job initialization launcher. Allowed choices are `none`, `pytorch`, `slurm`, `mpi`. Especially, if set to none, it will test in a non-distributed mode.
- `LOCAL_RANK`: ID for local rank. If not specified, it will be set to 0.

Examples:

Assume that you have already downloaded the checkpoints to the directory `checkpoints/`.

1. Test TSN on Kinetics-400 (without saving the test results) and evaluate the top-k accuracy and mean class accuracy.

    ```shell
    python tools/test.py configs/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb.py \
        checkpoints/SOME_CHECKPOINT.pth \
        --eval top_k_accuracy mean_class_accuracy
    ```

2. Test TSN on Something-Something V1 with 8 GPUS, and evaluate the top-k accuracy.

    ```shell
    ./tools/dist_test.sh configs/recognition/tsn/tsn_r50_1x1x8_50e_sthv1_rgb.py \
        checkpoints/SOME_CHECKPOINT.pth \
        8 --out results.pkl --eval top_k_accuracy
    ```

3. Test TSN on Kinetics-400 in slurm environment and evaluate the top-k accuracy

    ```shell
    python tools/test.py configs/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb.py \
        checkpoints/SOME_CHECKPOINT.pth \
        --launcher slurm --eval top_k_accuracy
    ```


### High-level APIs for testing a video and rawframes.

Here is an example of building the model and testing a given video.

```python
import torch

from mmaction.apis import init_recognizer, inference_recognizer

config_file = 'configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
checkpoint_file = 'checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth'

# assign the desired device.
device = 'cuda:0' # or 'cpu'
device = torch.device(device)

 # build the model from a config file and a checkpoint file
model = init_recognizer(config_file, checkpoint_file, device=device)

# test a single video and show the result:
video = 'demo/demo.mp4'
labels = 'demo/label_map.txt'
results = inference_recognizer(model, video, labels)

# show the results
print(f'The top-5 labels with corresponding scores are:')
for result in results:
    print(f'{result[0]}: ', result[1])
```

Here is an example of building the model and testing with a given rawframes directory.

```python
import torch

from mmaction.apis import init_recognizer, inference_recognizer

config_file = 'configs/recognition/tsn/tsn_r50_inference_1x1x3_100e_kinetics400_rgb.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
checkpoint_file = 'checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth'

# assign the desired device.
device = 'cuda:0' # or 'cpu'
device = torch.device(device)

 # build the model from a config file and a checkpoint file
model = init_recognizer(config_file, checkpoint_file, device=device, use_frames=True)

# test rawframe directory of a single video and show the result:
video = 'SOME_DIR_PATH/'
labels = 'demo/label_map.txt'
results = inference_recognizer(model, video, labels, use_frames=True)

# show the results
print(f'The top-5 labels with corresponding scores are:')
for result in results:
    print(f'{result[0]}: ', result[1])
```

Here is an example of building the model and testing with a given video url.

```python
import torch

from mmaction.apis import init_recognizer, inference_recognizer

config_file = 'configs/recognition/tsn/tsn_r50_inference_1x1x3_100e_kinetics400_rgb.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
checkpoint_file = 'checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth'

# assign the desired device.
device = 'cuda:0' # or 'cpu'
device = torch.device(device)

 # build the model from a config file and a checkpoint file
model = init_recognizer(config_file, checkpoint_file, device=device, use_frames=True)

# test rawframe directory of a single video and show the result:
video = 'https://www.learningcontainer.com/wp-content/uploads/2020/05/sample-mp4-file.mp4'
labels = 'demo/label_map.txt'
results = inference_recognizer(model, video, labels, use_frames=True)

# show the results
print(f'The top-5 labels with corresponding scores are:')
for result in results:
    print(f'{result[0]}: ', result[1])
```

**Note**: We define `data_prefix` in config files and set it None as default for our provided inference configs.
If the `data_prefix` is not None, the path for the video file (or rawframe directory) to get will be `data_prefix/video`.
Here, the `video` is the param in the demo scripts above.
This detail can be found in `rawframe_dataset.py` and `video_dataset.py`. For example,

* When video (rawframes) path is `SOME_DIR_PATH/VIDEO.mp4` (`SOME_DIR_PATH/VIDEO_NAME/img_xxxxx.jpg`), and `data_prefix` is None in the config file,
the param `video` should be `SOME_DIR_PATH/VIDEO.mp4` (`SOME_DIR_PATH/VIDEO_NAME`).

* When video (rawframes) path is `SOME_DIR_PATH/VIDEO.mp4` (`SOME_DIR_PATH/VIDEO_NAME/img_xxxxx.jpg`), and `data_prefix` is `SOME_DIR_PATH` in the config file,
the param `video` should be `VIDEO.mp4` (`VIDEO_NAME`).

* When rawframes path is `VIDEO_NAME/img_xxxxx.jpg`, and `data_prefix` is None in the config file, the param `video` should be `VIDEO_NAME`.

* When passing a url instead of a local video file, you need to use OpenCV as the video decoding backend.

A notebook demo can be found in [demo/demo.ipynb](/demo/demo.ipynb)


## Build a Model

### Build a model with basic components

In MMAction2, model components are basically categorized as 4 types.

- recognizer: the whole recognizer model pipeline, usually contains a backbone and cls_head.
- backbone: usually an FCN network to extract feature maps, e.g., ResNet, BNInception.
- cls_head: the component for classification task, usually contains an FC layer with some pooling layers.
- localizer: the model for localization task, currently available: BSN, BMN.

Following some basic pipelines (e.g., `Recognizer2D`), the model structure
can be customized through config files with no pains.

If we want to implement some new components, e.g., the temporal shift backbone structure as
in [TSM: Temporal Shift Module for Efficient Video Understanding](https://arxiv.org/abs/1811.08383), there are several things to do.

1. create a new file in `mmaction/models/backbones/resnet_tsm.py`.

    ```python
    from ..registry import BACKBONES
    from .resnet import ResNet

    @BACKBONES.register_module()
    class ResNetTSM(ResNet):

      def __init__(self,
                   depth,
                   num_segments=8,
                   is_shift=True,
                   shift_div=8,
                   shift_place='blockres',
                   temporal_pool=False,
                   **kwargs):
          pass

      def forward(self, x):
          # implementation is ignored
          pass
    ```

2. Import the module in `mmaction/models/backbones/__init__.py`
    ```python
    from .resnet_tsm import ResNetTSM
    ```

3. modify the config file from

    ```python
    backbone=dict(
      type='ResNet',
      pretrained='torchvision://resnet50',
      depth=50,
      norm_eval=False)
    ```
   to
    ```python
    backbone=dict(
        type='ResNetTSM',
        pretrained='torchvision://resnet50',
        depth=50,
        norm_eval=False,
        shift_div=8)
    ```

### Write a new model

To write a new recognition pipeline, you need to inherit from `BaseRecognizer`,
which defines the following abstract methods.

- `forward_train()`: forward method of the training mode.
- `forward_test()`: forward method of the testing mode.

[Recognizer2D](/mmaction/models/recognizers/recognizer2d.py) and [Recognizer3D](/mmaction/models/recognizers/recognizer3d.py)
are good examples which show how to do that.


## Train a Model

### Iteration pipeline

MMAction2 implements distributed training and non-distributed training,
which uses `MMDistributedDataParallel` and `MMDataParallel` respectively.

We adopt distributed training for both single machine and multiple machines.
Supposing that the server has 8 GPUs, 8 processes will be started and each process runs on a single GPU.

Each process keeps an isolated model, data loader, and optimizer.
Model parameters are only synchronized once at the beginning.
After a forward and backward pass, gradients will be allreduced among all GPUs,
and the optimizer will update model parameters.
Since the gradients are allreduced, the model parameter stays the same for all processes after the iteration.

### Training setting

All outputs (log files and checkpoints) will be saved to the working directory,
which is specified by `work_dir` in the config file.

By default we evaluate the model on the validation set after each epoch, you can change the evaluation interval by modifying the interval argument in the training config
```python
evaluation = dict(interval=5)  # This evaluate the model per 5 epoch.
```

According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you need to set the learning rate proportional to the batch size if you use different GPUs or videos per GPU, e.g., lr=0.01 for 4 GPUs * 2 video/gpu and lr=0.08 for 16 GPUs * 4 video/gpu.

### Train with a single GPU

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

If you want to specify the working directory in the command, you can add an argument `--work-dir ${YOUR_WORK_DIR}`.

### Train with multiple GPUs

```shell
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

Optional arguments are:

- `--validate` (**strongly recommended**): Perform evaluation at every k (default value is 5, which can be modified by changing the `interval` value in `evaluation` dict in each config file) epochs during the training.
- `--work-dir ${WORK_DIR}`: Override the working directory specified in the config file.
- `--resume-from ${CHECKPOINT_FILE}`: Resume from a previous checkpoint file.
- `--gpus ${GPU_NUM}`: Number of gpus to use, which is only applicable to non-distributed training.
- `--gpu-ids ${GPU_IDS}`: IDs of gpus to use, which is only applicable to non-distributed training.
- `--seed ${SEED}`: Seed id for random state in python, numpy and pytorch to generate random numbers.
- `--deterministic`: If specified, it will set deterministic options for CUDNN backend.
- `JOB_LAUNCHER`: Items for distributed job initialization launcher. Allowed choices are `none`, `pytorch`, `slurm`, `mpi`. Especially, if set to none, it will test in a non-distributed mode.
- `LOCAL_RANK`: ID for local rank. If not specified, it will be set to 0.

Difference between `resume-from` and `load-from`:
`resume-from` loads both the model weights and optimizer status, and the epoch is also inherited from the specified checkpoint. It is usually used for resuming the training process that is interrupted accidentally.
`load-from` only loads the model weights and the training epoch starts from 0. It is usually used for finetuning.

Here is an example of using 8 GPUs to load TSN checkpoint.

```shell
./tools/dist_train.sh configs/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb.py 8 --resume-from work_dirs/tsn_r50_1x1x3_100e_kinetics400_rgb/latest.pth
```

### Train with multiple machines

If you can run MMAction2 on a cluster managed with [slurm](https://slurm.schedmd.com/), you can use the script `slurm_train.sh`. (This script also supports single machine training.)

```shell
[GPUS=${GPUS}] ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} [--work-dir ${WORK_DIR}]
```

Here is an example of using 16 GPUs to train TSN on the dev partition in a slurm cluster. (use `GPUS_PER_NODE=8` to specify a single slurm cluster node with 8 GPUs.)

```shell
GPUS=16 ./tools/slurm_train.sh dev tsn_r50_k400 configs/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb.py --work-dir work_dirs/tsn_r50_1x1x3_100e_kinetics400_rgb
```

You can check [slurm_train.sh](/tools/slurm_train.sh) for full arguments and environment variables.

If you have just multiple machines connected with ethernet, you can refer to
pytorch [launch utility](https://pytorch.org/docs/stable/distributed.html#launch-utility).
Usually it is slow if you do not have high speed networking like InfiniBand.

### Launch multiple jobs on a single machine

If you launch multiple jobs on a single machine, e.g., 2 jobs of 4-GPU training on a machine with 8 GPUs,
you need to specify different ports (29500 by default) for each job to avoid communication conflict.

If you use `dist_train.sh` to launch training jobs, you can set the port in commands.

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_train.sh ${CONFIG_FILE} 4
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 ./tools/dist_train.sh ${CONFIG_FILE} 4
```

If you use launch training jobs with slurm, you need to modify the config files (usually the 6th line from the bottom in config files) to set different communication ports.

In `config1.py`,
```python
dist_params = dict(backend='nccl', port=29500)
```

In `config2.py`,
```python
dist_params = dict(backend='nccl', port=29501)
```

Then you can launch two jobs with `config1.py` ang `config2.py`.

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=4 ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config1.py [--work-dir ${WORK_DIR}]
CUDA_VISIBLE_DEVICES=4,5,6,7 GPUS=4 ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config2.py [--work-dir ${WORK_DIR}]
```

## Tutorials

Currently, we provide some tutorials for users to [finetune model](tutorials/finetune.md),
[add new dataset](tutorials/new_dataset.md), [customize data pipelines](tutorials/data_pipeline.md),
[add new modules](tutorials/new_modules.md), [export a model to ONNX](tutorials/export_model.md) and [customize runtime settings](tutorials/customize_runtime.md).
