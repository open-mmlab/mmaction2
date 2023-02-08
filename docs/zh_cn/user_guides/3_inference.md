# 教程3：利用现有模型进行推理

MMAction2 在 [Model Zoo](../modelzoo.md) 中提供预训练的视频理解模型。
本教程将展示**如何使用现有模型对给定视频进行推理**。

至于如何在标准数据集上测试现有模型，请参阅这该[指南](./4_train_test.md#test)

## 给定视频的推理

MMAction2提供了高级 Python APIs，用于对给定视频进行推理:

- [init_recognizer](mmaction.apis.init_recognizer): 用配置和检查点初始化一个识别器。
- [inference_recognizer](mmaction.apis.inference_recognizer): 对给定视频进行推理。

下面是一个使用 Kinetics-400 预训练检查点在给定视频上构建模型和推理的示例。

```{note}
如果使用mmaction2作为第三方包，则需要下载示例中的config和演示视频。

下载所需的配置：'mim download mmaction2 --config tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb --dest .'

下载所需的演示视频：'wget https://github.com/open-mmlab/mmaction2/blob/dev-1.x/demo/demo.mp4'
```

```python
from mmaction.apis import inference_recognizer, init_recognizer

config_path = 'configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb.py'
checkpoint_path = 'https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb_20220906-2692d16c.pth' # 可以是本地路径
img_path = 'demo/demo.mp4'   # 您可以指定自己的视频路径

# 从配置文件和检查点文件构建模型
model = init_recognizer(config_path, checkpoint_path, device="cpu")  # 也可以是 'cuda:0'
# 测试单个视频
result = inference_recognizer(model, img_path)
```

`result` 是一个包含 `pred_scores` 的字典。动作识别示例代码详见 [demo/demo.py](https://github.com/open-mmlab/mmaction2/blob/1.x/demo/demo.py)。
