Apart from training/testing scripts, We provide lots of useful tools under the `tools/` directory.

## Useful Tools Link

<!-- TOC -->

- [Log Analysis](#log-analysis)
- [Model Complexity](#model-complexity)
- [Model Conversion](#model-conversion)
  - [MMAction2 model to ONNX (experimental)](#mmaction2-model-to-onnx--experimental-)
  - [Prepare a model for publishing](#prepare-a-model-for-publishing)
- [Miscellaneous](#miscellaneous)
  - [Evaluating a metric](#evaluating-a-metric)
  - [Print the entire config](#print-the-entire-config)

<!-- TOC -->

## Log Analysis

`tools/analysis/analyze_logs.py` plots loss/top-k acc curves given a training log file. Run `pip install seaborn` first to install the dependency.

![acc_curve_image](imgs/acc_curve.png)

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

**Note**: This tool is still experimental and we do not guarantee that the number is absolutely correct.
You may use the result for simple comparisons, but double check it before you adopt it in technical reports or papers.

(1) FLOPs are related to the input shape while parameters are not. The default input shape is (1, 3, 340, 256) for 2D recognizer, (1, 3, 32, 340, 256) for 3D recognizer.
(2) Some operators are not counted into FLOPs like GN and custom operators. Refer to [`mmcv.cnn.get_model_complexity_info()`](https://github.com/open-mmlab/mmcv/blob/master/mmcv/cnn/utils/flops_counter.py) for details.

## Model Conversion

### MMAction2 model to ONNX (experimental)

`/tools/pytorch2onnx.py` is a script to convert model to [ONNX](https://github.com/onnx/onnx) format.
It also supports comparing the output results between Pytorch and ONNX model for verification.
Run `pip install onnx onnxruntime` first to install the dependency.

- For recognizers, please run:

    ```shell
    python tools/pytorch2onnx.py $CONFIG_PATH $CHECKPOINT_PATH --shape $SHAPE --verify
    ```

- For localizers, please run:

    ```shell
    python tools/pytorch2onnx.py $CONFIG_PATH $CHECKPOINT_PATH --is-localizer --shape $SHAPE --verify
    ```

### Prepare a model for publishing

`tools/publish_model.py` helps users to prepare their model for publishing.

Before you upload a model to AWS, you may want to:

(1) convert model weights to CPU tensors.
(2) delete the optimizer states.
(3) compute the hash of the checkpoint file and append the hash id to the filename.

```shell
python tools/publish_model.py ${INPUT_FILENAME} ${OUTPUT_FILENAME}
```

E.g.,

```shell
python tools/publish_model.py work_dirs/tsn_r50_1x1x3_100e_kinetics400_rgb/latest.pth tsn_r50_1x1x3_100e_kinetics400_rgb.pth
```

The final output filename will be `tsn_r50_1x1x3_100e_kinetics400_rgb-{hash id}.pth`.

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
