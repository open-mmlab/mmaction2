# Useful Tools

Apart from training/testing scripts, We provide lots of useful tools under the `tools/` directory.

## Useful Tools Link

<!-- TOC -->

- [Useful Tools](#useful-tools)
  - [Useful Tools Link](#useful-tools-link)
  - [Model Conversion](#model-conversion)
    - [Prepare a model for publishing](#prepare-a-model-for-publishing)
  - [Miscellaneous](#miscellaneous)
    - [Evaluating a metric](#evaluating-a-metric)
    - [Print the entire config](#print-the-entire-config)
    - [Check videos](#check-videos)
    - [Multi-Stream Fusion](#multi-stream-fusion)

<!-- TOC -->

## Model Conversion

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
python tools/deployment/publish_model.py work_dirs/tsn_r50_8xb32-1x1x3-100e_kinetics400-rgb/latest.pth tsn_r50_1x1x3_100e_kinetics400_rgb.pth
```

The final output filename will be `tsn_r50_8xb32-1x1x3-100e_kinetics400-rgb-{hash id}.pth`.

## Miscellaneous

### Evaluating a metric

`tools/analysis_tools/eval_metric.py` evaluates certain metrics of the results saved in a file according to a config file.

The saved result file is created on `tools/test.py` by setting the arguments `--out ${RESULT_FILE}` to indicate the result file,
which stores the final output of the whole model.

```shell
python tools/analysis/eval_metric.py ${CONFIG_FILE} ${RESULT_FILE} [--eval ${EVAL_METRICS}] [--cfg-options ${CFG_OPTIONS}] [--eval-options ${EVAL_OPTIONS}]
```

### Print the entire config

`tools/analysis_tools/print_config.py` prints the whole config verbatim, expanding all its imports.

```shell
python tools/analysis_tools/print_config.py ${CONFIG} [-h] [--options ${OPTIONS [OPTIONS...]}]
```

### Check videos

`tools/analysis_tools/check_videos.py` uses specified video encoder to iterate all samples that are specified by the input configuration file, looks for invalid videos (corrupted or missing), and saves the corresponding file path to the output file. Please note that after deleting invalid videos, users need to regenerate the video file list.

```shell
python tools/analysis_tools/check_videos.py ${CONFIG} [-h] [--options OPTIONS [OPTIONS ...]] [--cfg-options CFG_OPTIONS [CFG_OPTIONS ...]] [--output-file OUTPUT_FILE] [--split SPLIT] [--decoder DECODER] [--num-processes NUM_PROCESSES] [--remove-corrupted-videos]
```

### Multi-Stream Fusion

`tools/analysis_tools/report_accuracy.py` uses the dumped results (by setting `--dump res.pkl` when testing) to fuse the multi-stream prediction scores, i.e., late fusion.

```shell
python tools/analysis_tools/report_accuracy.py [--preds ${RESULT_PKL_1 [RESULT_PKL_2 ...]}] [--coefficients ${COEFFICIENT_1 [COEFFICIENT_2, ...]}] [--apply-softmax]
```

Take joint-bone fusion as an example, which is a general practice in the task of skeleton-based action recognition.

```shell
python tools/analysis_tools/report_accuracy.py --preds demo/fuse/joint.pkl demo/fuse/bone.pkl --coefficients 1.0 1.0
```

```
Mean Class Accuracy: 0.9180
Top 1 Accuracy: 0.9333
Top 5 Accuracy: 0.9833
```
