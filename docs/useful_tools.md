Apart from training/testing scripts, We provide lots of useful tools under the `tools/` directory.

## Useful Tools Link

<!-- TOC -->

- [Useful Tools Link](#useful-tools-link)
- [Log Analysis](#log-analysis)
- [Model Complexity](#model-complexity)
- [Model Conversion](#model-conversion)
  - [MMAction2 model to ONNX (experimental)](#mmaction2-model-to-onnx-experimental)
  - [Prepare a model for publishing](#prepare-a-model-for-publishing)
- [Model Serving](#model-serving)
  - [1. Convert model from MMAction2 to TorchServe](#1-convert-model-from-mmaction2-to-torchserve)
  - [2. Build `mmaction-serve` docker image](#2-build-mmaction-serve-docker-image)
  - [3. Launch `mmaction-serve`](#3-launch-mmaction-serve)
  - [4. Test deployment](#4-test-deployment)
- [Miscellaneous](#miscellaneous)
  - [Evaluating a metric](#evaluating-a-metric)
  - [Print the entire config](#print-the-entire-config)
  - [Check videos](#check-videos)

<!-- TOC -->

## Log Analysis

`tools/analysis/analyze_logs.py` plots loss/top-k acc curves given a training log file. Run `pip install seaborn` first to install the dependency.

![acc_curve_image](https://github.com/open-mmlab/mmaction2/raw/master/resources/acc_curve.png)

```shell
python tools/analysis/analyze_logs.py plot_curve ${JSON_LOGS} [--keys ${KEYS}] [--title ${TITLE}] [--legend ${LEGEND}] [--backend ${BACKEND}] [--style ${STYLE}] [--out ${OUT_FILE}]
```

Examples:

- Plot the classification loss of some run.

    ```shell
    python tools/analysis/analyze_logs.py plot_curve log.json --keys loss_cls --legend loss_cls
    ```

- Plot the top-1 acc and top-5 acc of some run, and save the figure to a pdf.

    ```shell
    python tools/analysis/analyze_logs.py plot_curve log.json --keys top1_acc top5_acc --out results.pdf
    ```

- Compare the top-1 acc of two runs in the same figure.

    ```shell
    python tools/analysis/analyze_logs.py plot_curve log1.json log2.json --keys top1_acc --legend run1 run2
    ```

    You can also compute the average training speed.

    ```shell
    python tools/analysis/analyze_logs.py cal_train_time ${JSON_LOGS} [--include-outliers]
    ```

- Compute the average training speed for a config file.

    ```shell
    python tools/analysis/analyze_logs.py cal_train_time work_dirs/some_exp/20200422_153324.log.json
    ```

    The output is expected to be like the following.

    ```text
    -----Analyze train time of work_dirs/some_exp/20200422_153324.log.json-----
    slowest epoch 60, average time is 0.9736
    fastest epoch 18, average time is 0.9001
    time std over epochs is 0.0177
    average iter time: 0.9330 s/iter
    ```

## Model Complexity

`/tools/analysis/get_flops.py` is a script adapted from [flops-counter.pytorch](https://github.com/sovrasov/flops-counter.pytorch) to compute the FLOPs and params of a given model.

```shell
python tools/analysis/get_flops.py ${CONFIG_FILE} [--shape ${INPUT_SHAPE}]
```

We will get the result like this

```text
==============================
Input shape: (1, 3, 32, 340, 256)
Flops: 37.1 GMac
Params: 28.04 M
==============================
```

:::{note}
This tool is still experimental and we do not guarantee that the number is absolutely correct.
You may use the result for simple comparisons, but double check it before you adopt it in technical reports or papers.

(1) FLOPs are related to the input shape while parameters are not. The default input shape is (1, 3, 340, 256) for 2D recognizer, (1, 3, 32, 340, 256) for 3D recognizer.
(2) Some operators are not counted into FLOPs like GN and custom operators. Refer to [`mmcv.cnn.get_model_complexity_info()`](https://github.com/open-mmlab/mmcv/blob/master/mmcv/cnn/utils/flops_counter.py) for details.
:::

## Model Conversion

### MMAction2 model to ONNX (experimental)

`/tools/deployment/pytorch2onnx.py` is a script to convert model to [ONNX](https://github.com/onnx/onnx) format.
It also supports comparing the output results between Pytorch and ONNX model for verification.
Run `pip install onnx onnxruntime` first to install the dependency.
Please note that a softmax layer could be added for recognizers by `--softmax` option, in order to get predictions in range `[0, 1]`.

- For recognizers, please run:

    ```shell
    python tools/deployment/pytorch2onnx.py $CONFIG_PATH $CHECKPOINT_PATH --shape $SHAPE --verify
    ```

- For localizers, please run:

    ```shell
    python tools/deployment/pytorch2onnx.py $CONFIG_PATH $CHECKPOINT_PATH --is-localizer --shape $SHAPE --verify
    ```

### Prepare a model for publishing

`tools/deployment/publish_model.py` helps users to prepare their model for publishing.

Before you upload a model to AWS, you may want to:

(1) convert model weights to CPU tensors.
(2) delete the optimizer states.
(3) compute the hash of the checkpoint file and append the hash id to the filename.

```shell
python tools/deployment/publish_model.py ${INPUT_FILENAME} ${OUTPUT_FILENAME}
```

E.g.,

```shell
python tools/deployment/publish_model.py work_dirs/tsn_r50_1x1x3_100e_kinetics400_rgb/latest.pth tsn_r50_1x1x3_100e_kinetics400_rgb.pth
```

The final output filename will be `tsn_r50_1x1x3_100e_kinetics400_rgb-{hash id}.pth`.

## Model Serving

In order to serve an `MMAction2` model with [`TorchServe`](https://pytorch.org/serve/), you can follow the steps:

### 1. Convert model from MMAction2 to TorchServe

```shell
python tools/deployment/mmaction2torchserve.py ${CONFIG_FILE} ${CHECKPOINT_FILE} \
--output_folder ${MODEL_STORE} \
--model-name ${MODEL_NAME} \
--label-file ${LABLE_FILE}

```

### 2. Build `mmaction-serve` docker image

```shell
DOCKER_BUILDKIT=1 docker build -t mmaction-serve:latest docker/serve/
```

### 3. Launch `mmaction-serve`

Check the official docs for [running TorchServe with docker](https://github.com/pytorch/serve/blob/master/docker/README.md#running-torchserve-in-a-production-docker-environment).

Example:

```shell
docker run --rm \
--cpus 8 \
--gpus device=0 \
-p8080:8080 -p8081:8081 -p8082:8082 \
--mount type=bind,source=$MODEL_STORE,target=/home/model-server/model-store \
mmaction-serve:latest
```

**Note**: ${MODEL_STORE} needs to be an absolute path.
[Read the docs](https://github.com/pytorch/serve/blob/072f5d088cce9bb64b2a18af065886c9b01b317b/docs/rest_api.md) about the Inference (8080), Management (8081) and Metrics (8082) APis

### 4. Test deployment

```shell
# Assume you are under the directory `mmaction2`
curl http://127.0.0.1:8080/predictions/${MODEL_NAME} -T demo/demo.mp4
```

You should obtain a response similar to:

```json
{
  "arm wrestling": 1.0,
  "rock scissors paper": 4.962051880497143e-10,
  "shaking hands": 3.9761663406245873e-10,
  "massaging feet": 1.1924419784925533e-10,
  "stretching leg": 1.0601879096849842e-10
}
```

## Miscellaneous

### Evaluating a metric

`tools/analysis/eval_metric.py` evaluates certain metrics of the results saved in a file according to a config file.

The saved result file is created on `tools/test.py` by setting the arguments `--out ${RESULT_FILE}` to indicate the result file,
which stores the final output of the whole model.

```shell
python tools/analysis/eval_metric.py ${CONFIG_FILE} ${RESULT_FILE} [--eval ${EVAL_METRICS}] [--cfg-options ${CFG_OPTIONS}] [--eval-options ${EVAL_OPTIONS}]
```

### Print the entire config

`tools/analysis/print_config.py` prints the whole config verbatim, expanding all its imports.

```shell
python tools/print_config.py ${CONFIG} [-h] [--options ${OPTIONS [OPTIONS...]}]
```

### Check videos

`tools/analysis/check_videos.py` uses specified video encoder to iterate all samples that are specified by the input configuration file, looks for invalid videos (corrupted or missing), and saves the corresponding file path to the output file. Please note that after deleting invalid videos, users need to regenerate the video file list.

```shell
python tools/analysis/check_videos.py ${CONFIG} [-h] [--options OPTIONS [OPTIONS ...]] [--cfg-options CFG_OPTIONS [CFG_OPTIONS ...]] [--output-file OUTPUT_FILE] [--split SPLIT] [--decoder DECODER] [--num-processes NUM_PROCESSES] [--remove-corrupted-videos]
```
