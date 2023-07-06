# 使用现有模型进行推理

MMAction2 在[模型库](../modelzoo.md)中提供了预训练的视频理解模型。本文将展示如何使用现有模型对给定的视频进行推理。

关于如何在标准数据集上测试现有模型，请参考这个[指南](./train_test.md#test)。

## 对给定视频进行推理

MMAction2 提供了用于对给定视频进行推理的高级 Python API：

- [init_recognizer](mmaction.apis.init_recognizer): 使用配置文件和权重文件初始化一个识别器
- [inference_recognizer](mmaction.apis.inference_recognizer): 对给定视频进行推理

下面是一个使用 Kinitics-400 预训练权重构建模型并对给定视频进行推理的示例。

```{note}
如果您将 mmaction2 用作第三方包，您需要下载示例中的配置文件和演示视频。

运行 'mim download mmaction2 --config tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb --dest .' 下载所需的配置文件。

运行 'wget https://github.com/open-mmlab/mmaction2/blob/main/demo/demo.mp4' 下载所需的演示视频。
```

```python
from mmaction.apis import inference_recognizer, init_recognizer

config_path = 'configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb.py'
checkpoint_path = 'https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb_20220906-2692d16c.pth' # 可以是本地路径
img_path = 'demo/demo.mp4'   # 您可以指定自己的图片路径

# 从配置文件和权重文件中构建模型
model = init_recognizer(config_path, checkpoint_path, device="cpu")  # device 可以是 'cuda:0'
# 对单个视频进行测试
result = inference_recognizer(model, img_path)
```

`result` 是一个包含 `pred_scores` 的字典。

示例中的动作识别演示可以在[demo/demo.py](https://github.com/open-mmlab/mmaction2/blob/main/demo/demo.py)中找到。
