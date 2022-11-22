# MMAction2 Deployment

- [MMAction2 Deployment](#mmaction2-deployment)
  - [Installation](#installation)
    - [Install mmaction2](#install-mmaction2)
    - [Install mmdeploy](#install-mmdeploy)
  - [Convert model](#convert-model)
    - [Convert video recognition model](#convert-video-recognition-model)
  - [Model specification](#model-specification)
  - [Model Inference](#model-inference)
    - [Backend model inference](#backend-model-inference)
    - [SDK model inference](#sdk-model-inference)
      - [Video recognition SDK model inference](#video-recognition-sdk-model-inference)
  - [Supported models](#supported-models)

______________________________________________________________________

The deployment of OpenMMLab codebases, including MMAction2, MMDetection and so on are supported by [MMDeploy](https://github.com/open-mmlab/mmdeploy).
The latest deployment guide for MMAction2 can be found from [here](https://mmdeploy.readthedocs.io/en/dev-1.x/04-supported-codebases/mmdet.html).

## Installation

### Install mmaction2

Please follow the [installation guide](https://github.com/open-mmlab/mmaction2#installation) to install mmocr.

### Install mmdeploy

There are several methods to install mmdeploy, among which you can choose an appropriate one according to your target platform and device.

**Method I：** Install precompiled package

You can download the latest release package from [here](https://github.com/open-mmlab/mmdeploy/releases)

**Method II：** Build using scripts

If your target platform is **Ubuntu 18.04 or later version**, we encourage you to run
[scripts](../01-how-to-build/build_from_script.md). For example, the following commands install mmdeploy as well as inference engine - `ONNX Runtime`.

```shell
git clone --recursive https://github.com/open-mmlab/mmdeploy.git
cd mmdeploy
python3 tools/scripts/build_ubuntu_x64_ort.py $(nproc)
export PYTHONPATH=$(pwd)/build/lib:$PYTHONPATH
export LD_LIBRARY_PATH=$(pwd)/../mmdeploy-dep/onnxruntime-linux-x64-1.8.1/lib/:$LD_LIBRARY_PATH
```

**Method III:** Build from source

If neither **I** nor **II** meets your requirements, [building mmdeploy from source](../01-how-to-build/build_from_source.md) is the last option.

## Convert model

You can use [tools/deploy.py](https://github.com/open-mmlab/mmdeploy/blob/master/tools/deploy.py) to convert mmocr models to the specified backend models. Its detailed usage can be learned from [here](https://github.com/open-mmlab/mmdeploy/blob/master/docs/en/02-how-to-run/convert_model.md#usage).

When using `tools/deploy.py`, it is crucial to specify the correct deployment config. We've already provided builtin deployment config [files](https://github.com/open-mmlab/mmdeploy/tree/master/configs/mmaction) of all supported backends for mmocr, under which the config file path follows the pattern:

```
{task}/{task}_{backend}-{precision}_{static | dynamic}_{shape}.py
```

其中：

- **{task}:** task in mmaction2.
- **{backend}:** inference backend, such as onnxruntime, tensorrt, pplnn, ncnn, openvino, coreml etc.
- **{precision}:** fp16, int8. When it's empty, it means fp32
- **{static | dynamic}:** static shape or dynamic shape
- **{shape}:** input shape or shape range of a model
- **{2d/3d}:** model type

In the next part，we will take `tsn` model from `video recognition` task as an example, showing how to convert them to onnx model that can be inferred by ONNX Runtime.

### Convert video recognition model

```shell
cd mmdeploy

# download tsn model from mmaction2 model zoo
mim download mmaction2 --config tsn_r50_1x1x3_100e_kinetics400_rgb --dest .

# convert mmaction2 model to onnxruntime model with dynamic shape
python tools/deploy.py \
    configs/mmaction/video-recognition/video-recognition_2d_onnxruntime_static.py \
    tsn_r50_1x1x3_100e_kinetics400_rgb.py \
    tsn_r50_256p_1x1x3_100e_kinetics400_rgb_20200725-22592236.pth \
    tests/data/arm_wrestling.mp4 \
    --work-dir mmdeploy_models/mmaction/tsn/ort \
    --device cpu \
    --show \
    --dump-info
```

## Model specification

Before moving on to model inference chapter, let's know more about the converted model structure which is very important for model inference.

The converted model locates in the working directory like `mmdeploy_models/mmaction/tsn/ort` in the previous example. It includes:

```
mmdeploy_models/mmaction/tsn/ort
├── deploy.json
├── detail.json
├── end2end.onnx
└── pipeline.json
```

in which,

- **end2end.onnx**: backend model which can be inferred by ONNX Runtime
- \***.json**: the necessary information for mmdeploy SDK

The whole package **mmdeploy_models/mmocr/dbnet/ort** is defined as **mmdeploy SDK model**, i.e., **mmdeploy SDK model** includes both backend model and inference meta information.

## Model Inference

### Backend model inference

Take the previous converted `end2end.onnx` mode of `tsn` as an example, you can use the following code to inference the model and visualize the results.

```python
from mmdeploy.apis.utils import build_task_processor
from mmdeploy.utils import get_input_shape, load_config
import numpy as np
import torch

deploy_cfg = 'configs/mmaction/video-recognition/video-recognition_2d_onnxruntime_static.py'
model_cfg = 'tsn_r50_1x1x3_100e_kinetics400_rgb.py'
device = 'cpu'
backend_model = ['./mmdeploy_models/mmaction2/tsn/ort/end2end.onnx']
image = 'tests/data/arm_wrestling.mp4'

# read deploy_cfg and model_cfg
deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)

# build task and backend model
task_processor = build_task_processor(model_cfg, deploy_cfg, device)
model = task_processor.init_backend_model(backend_model)

# process input image
input_shape = get_input_shape(deploy_cfg)
model_inputs, _ = task_processor.create_input(image, input_shape)

# do model inference
with torch.no_grad():
    result = task_processor.run_inference(model, model_inputs)

# show top5-results
result = np.array(result[0])
top_index = np.argsort(result)[::-1]
for i in range(5):
    index = top_index[i]
    print(index, result[index])
```

### SDK model inference

Given the above SDK model of `tsn` you can also perform SDK model inference like following,

#### Video recognition SDK model inference

```python
from mmdeploy_python import VideoRecognizer
import cv2

# refer to demo/python/video_recognition.py
# def SampleFrames(cap, clip_len, frame_interval, num_clips):
#  ...

cap = cv2.VideoCapture('tests/data/arm_wrestling.mp4')

clips, info = SampleFrames(cap, 1, 1, 25)

# create a recognizer
recognizer = VideoRecognizer(model_path='./mmdeploy_models/mmaction/tsn/ort', device_name='cpu', device_id=0)
# perform inference
result = recognizer(clips, info)
# show inference result
for label_id, score in result:
    print(label_id, score)
```

Besides python API, mmdeploy SDK also provides other FFI (Foreign Function Interface), such as C, C++, C#, Java and so on. You can learn their usage from [demos](https://github.com/open-mmlab/mmdeploy/demo).

> MMAction2 only API of c, c++ and python for now.

## Supported models

Please refer to here\[https://github.com/open-mmlab/mmdeploy/blob/master/docs/en/04-supported-codebases/mmaction2.md#supported-models\] for the supported model list
