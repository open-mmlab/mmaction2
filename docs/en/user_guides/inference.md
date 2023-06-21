# Inference with existing models

MMAction2 provides pre-trained models for video understanding in [Model Zoo](../modelzoo.md).
This note will show **how to use existing models to inference on given video**.

As for how to test existing models on standard datasets, please see this [guide](./train_test.md#test)

## Inference on a given video

MMAction2 provides high-level Python APIs for inference on a given video:

- [init_recognizer](mmaction.apis.init_recognizer): Initialize a recognizer with a config and checkpoint
- [inference_recognizer](mmaction.apis.inference_recognizer): Inference on a given video

Here is an example of building the model and inference on a given video by using Kinitics-400 pre-trained checkpoint.

```{note}
If you use mmaction2 as a 3rd-party package, you need to download the conifg and the demo video in the example.

Run 'mim download mmaction2 --config tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb --dest .' to download the required config.

Run 'wget https://github.com/open-mmlab/mmaction2/blob/main/demo/demo.mp4' to download the desired demo video.
```

```python
from mmaction.apis import inference_recognizer, init_recognizer

config_path = 'configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb.py'
checkpoint_path = 'https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb_20220906-2692d16c.pth' # can be a local path
img_path = 'demo/demo.mp4'   # you can specify your own picture path

# build the model from a config file and a checkpoint file
model = init_recognizer(config_path, checkpoint_path, device="cpu")  # device can be 'cuda:0'
# test a single image
result = inference_recognizer(model, img_path)
```

`result` is a dictionary containing `pred_scores`.

An action recognition demo can be found in [demo/demo.py](https://github.com/open-mmlab/mmaction2/blob/main/demo/demo.py).
