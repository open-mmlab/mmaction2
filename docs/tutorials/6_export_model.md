# Tutorial 6: Exporting a model to ONNX

Open Neural Network Exchange [(ONNX)](https://onnx.ai/) is an open ecosystem that empowers AI developers to choose the right tools as their project evolves.

<!-- TOC -->

- [Supported Models](#supported-models)
- [Usage](#usage)
  - [Prerequisite](#prerequisite)
  - [Recognizers](#recognizers)
  - [Localizers](#localizers)

<!-- TOC -->

## Supported Models

So far, our codebase supports onnx exporting from pytorch models trained with MMAction2. The supported models are:

- I3D
- TSN
- TIN
- TSM
- R(2+1)D
- SLOWFAST
- SLOWONLY
- BMN
- BSN(tem, pem)

## Usage

For simple exporting, you can use the [script](/tools/deployment/pytorch2onnx.py) here. Note that the package `onnx` and `onnxruntime` are required for verification after exporting.

### Prerequisite

First, install onnx.

```shell
pip install onnx onnxruntime
```

We provide a python script to export the pytorch model trained by MMAction2 to ONNX.

```shell
python tools/deployment/pytorch2onnx.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--shape ${SHAPE}] \
    [--verify] [--show] [--output-file ${OUTPUT_FILE}]  [--is-localizer] [--opset-version ${VERSION}]
```

Optional arguments:

- `--shape`: The shape of input tensor to the model. For 2D recognizer(e.g. TSN), the input should be `$batch $clip $channel $height $width`(e.g. `1 1 3 224 224`); For 3D recognizer(e.g. I3D), the input should be `$batch $clip $channel $time $height $width`(e.g. `1 1 3 32 224 224`); For localizer such as BSN, the input for each module is different, please check the `forward` function for it. If not specified, it will be set to `1 1 3 224 224`.
- `--verify`: Determines whether to verify the exported model, runnably and numerically. If not specified, it will be set to `False`.
- `--show`: Determines whether to print the architecture of the exported model. If not specified, it will be set to `False`.
- `--output-file`: The output onnx model name. If not specified, it will be set to `tmp.onnx`.
- `--is-localizer`: Determines whether the model to be exported is a localizer. If not specified, it will be set to `False`.
- `--opset-version`: Determines the operation set version of onnx, we recommend you to use a higher version such as 11 for compatibility. If not specified, it will be set to `11`.
- `--softmax`: Determines whether to add a softmax layer at the end of recognizers. If not specified, it will be set to `False`. For now, localizers are not supported.

### Recognizers

For recognizers, please run:

```shell
python tools/deployment/pytorch2onnx.py $CONFIG_PATH $CHECKPOINT_PATH --shape $SHAPE --verify
```

### Localizers

For localizers, please run:

```shell
python tools/deployment/pytorch2onnx.py $CONFIG_PATH $CHECKPOINT_PATH --is-localizer --shape $SHAPE --verify
```

Please fire an issue if you discover any checkpoints that are not perfectly exported or suffer some loss in accuracy.
