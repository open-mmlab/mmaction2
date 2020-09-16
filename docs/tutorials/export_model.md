# Tutorial 5: Exporting a model to ONNX

Open Neural Network Exchange [(ONNX)](https://onnx.ai/) is an open ecosystem that empowers AI developers to choose the right tools as their project evolves. So far, our codebase supports onnx exporting from pytorch models trained with mmaction2. The supported models are:

+ I3D
+ TSN
+ TIN
+ TSM
+ R(2+1)D
+ SLOWFAST
+ SLOWONLY
+ BMN
+ BSN(tem, pem)

## Usage
For simple exporting, you can use the [script](../../tools/torch2onnx.py) here. Note that the package `onnx` is requried for verification after exporting.

### Prerequisite
First, install onnx.
```shell
pip install onnx
```

### Recognizers
For recognizers, if your model are trained with a config from mmaction2 and intend to inference it according to the test pipeline, simply run:
```shell
python tools/torch2onnx.py $CONFIG_PATH $CHECKPOINT_PATH
```

Otherwise, if you want to customize the input tensor shape, you can modify the `test_pipeline` in your config `$CONFIG_PATH`, or run:
```shell
python tools/torch2onnx.py $CONFIG_PATH $CHECKPOINT_PATH --input-size $BATCHS $CROPS $CHANNELS $CLIP_LENGTH $HEIGHT $WIDTH
```

### Localizer
For localizers, we *only* support customized input size, since our abstractions for localizers(eg. SSN, BMN) are not unified. Please run:
```shell
python tools/torch2onnx.py $CONFIG_PATH $CHECKPOINT_PATH --is-localizer --input-size $INPUT_SIZE
```

Please fire an issue if you discover any checkpoints that are not perfectly exported or suffer some loss in accuracy.
