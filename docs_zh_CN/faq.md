# 常见问题解答

本文这里列出了用户们遇到的一些常见问题，及相应的解决方案。
如果您发现了任何社区中经常出现的问题，也有了相应的解决方案，欢迎充实本文档来帮助他人。
如果本文档不包括您的问题，欢迎使用提供的 [模板](/.github/ISSUE_TEMPLATE/error-report.md) 创建问题，还请确保您在模板中填写了所有必需的信息。

## 安装

- **"No module named 'mmcv.ops'"; "No module named 'mmcv._ext'"**

    1. 使用 `pip uninstall mmcv` 卸载环境中已安装的 `mmcv`。
    2. 遵循 [MMCV 安装文档](https://mmcv.readthedocs.io/en/latest/#installation) 来安装 `mmcv-full`。

- **"OSError: MoviePy Error: creation of None failed because of the following error"**

    参照 [MMAction2 安装文档](https://github.com/open-mmlab/mmaction2/blob/master/docs_zh_CN/install.md#安装依赖包)
    1. 对于 Windows 用户，[ImageMagick](https://www.imagemagick.org/script/index.php) 不再被 MoviePy 自动检测，
    需要获取名为 `magick` 的 ImageMagick 二进制包的路径，来修改 `moviepy/config_defaults.py` 文件中的 `IMAGEMAGICK_BINARY`，如 `IMAGEMAGICK_BINARY = "C:\\Program Files\\ImageMagick_VERSION\\magick.exe"`
    2. 对于 Linux 用户，如果 ImageMagick 没有被 moviepy 检测，需要注释掉 `/etc/ImageMagick-6/policy.xml` 文件中的 `<policy domain="path" rights="none" pattern="@*" />`，即改为 `<!-- <policy domain="path" rights="none" pattern="@*" /> -->`。

## 数据

- **FileNotFound 如 `No such file or directory: xxx/xxx/img_00300.jpg`**

    在 MMAction2 中，对于帧数据集，`start_index` 的默认值为 1，而对于视频数据集， `start_index` 的默认值为 0。
    如果 FileNotFound 错误发生于视频的第一帧或最后一帧，则需根据视频首帧（即 `xxx_00000.jpg` 或 `xxx_00001.jpg`）的偏移量，修改配置文件中数据处理流水线的 `start_index` 值。

- **如何处理数据集中传入视频的尺寸？是把所有视频调整为固定尺寸，如 “340x256”，还是把所有视频的短边调整成相同的长度（256像素或320像素）？**

    从基准测试来看，总体来说，后者（把所有视频的短边调整成相同的长度）效果更好，所以“调整尺寸为短边256像素”被设置为默认的数据处理方式。用户可以在 [TSN 数据基准测试](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/tsn) 和 [SlowOnly 数据基准测试](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/tsn) 中查看相关的基准测试结果。

- **输入数据格式（视频或帧）与数据流水线不匹配，导致异常，如 `KeyError: 'total_frames'`**

    对于视频和帧，我们都有相应的流水线来处理。

    **对于视频**，应该在处理时首先对其进行解码。可选的解码方式，有 `DecordInit & DecordDecode`, `OpenCVInit & OpenCVDecode`, `PyAVInit & PyAVDecode` 等等。可以参照 [这个例子](https://github.com/open-mmlab/mmaction2/blob/023777cfd26bb175f85d78c455f6869673e0aa09/configs/recognition/slowfast/slowfast_r50_video_4x16x1_256e_kinetics400_rgb.py#L47-L49)。

    **对于帧**，已经事先在本地对其解码，所以使用 `RawFrameDecode` 对帧处理即可。可以参照 [这个例子](https://github.com/open-mmlab/mmaction2/blob/023777cfd26bb175f85d78c455f6869673e0aa09/configs/recognition/slowfast/slowfast_r50_8x8x1_256e_kinetics400_rgb.py#L49)。

    `KeyError: 'total_frames'` 是因为错误地使用了 `RawFrameDecode` 来处理视频。当输入是视频的时候，程序是无法事先得到 `total_frame` 的。

## 训练

- **如何使用训练过的识别器作为主干网络的预训练模型？**

    参照 [使用预训练模型](https://github.com/open-mmlab/mmaction2/blob/master/docs_zh_CN/tutorials/2_finetune.md#使用预训练模型)，
    如果想对整个网络使用预训练模型，可以在配置文件中，将 `load_from` 设置为预训练模型的链接。

    如果只想对主干网络使用预训练模型，可以在配置文件中，将主干网络 `backbone` 中的 `pretrained` 设置为预训练模型的地址或链接。
    在训练时，预训练模型中无法与主干网络对应的参数会被忽略。

- **如何实时绘制训练集和验证集的准确率/损失函数曲线图？**

    使用 `log_config` 中的 `TensorboardLoggerHook`，如：

    ```python
    log_config=dict(
        interval=20,
        hooks=[
            dict(type='TensorboardLoggerHook')
        ]
    )
    ```

    可以参照 [教程1：如何编写配置文件](tutorials/1_config.md)，[教程7：如何自定义模型运行参数](tutorials/7_customize_runtime.md#log-config)，和 [这个例子](https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/tsm/tsm_r50_1x1x8_50e_kinetics400_rgb.py#L118) 了解更多相关内容。

- **在 batchnorm.py 中抛出错误: Expected more than 1 value per channel when training**

    BatchNorm 层要求批大小（batch size）大于 1。构建数据集时， 若 `drop_last` 被设为 `False`，有时每个轮次的最后一个批次的批大小可能为 1，进而在训练时抛出错误，可以设置 `drop_last=True` 来避免该错误，如：

    ```python
    train_dataloader=dict(drop_last=True)
    ```

- **微调模型参数时，如何冻结主干网络中的部分参数？**

    可以参照 [`def _freeze_stages()`](https://github.com/open-mmlab/mmaction2/blob/0149a0e8c1e0380955db61680c0006626fd008e9/mmaction/models/backbones/x3d.py#L458) 和 [`frozen_stages`](https://github.com/open-mmlab/mmaction2/blob/0149a0e8c1e0380955db61680c0006626fd008e9/mmaction/models/backbones/x3d.py#L183-L184)。在分布式训练和测试时，还须设置 `find_unused_parameters = True`。

    实际上，除了少数模型，如 C3D 等，用户都能通过设置 `frozen_stages` 来冻结模型参数，因为大多数主干网络继承自 `ResNet` 和 `ResNet3D`，而这两个模型都支持 `_freeze_stages()` 方法。

- **如何在配置文件中设置 `load_from` 参数以进行模型微调？**

    MMAction2 在 `configs/_base_/default_runtime.py` 文件中将 `load_from=None` 设为默认。由于配置文件的可继承性，用户可直接在下游配置文件中设置 `load_from` 的值来进行更改。

## 测试

- **如何将预测分值用 softmax 归一化到 [0, 1] 区间内？**

    可以通过设置 `model['test_cfg'] = dict(average_clips='prob')` 来实现。

- **如果模型太大，连一个测试样例都没法放进显存，怎么办？**

    默认情况下，3D 模型是以 `10 clips x 3 crops` 的设置进行测试的，也即采样 10 个帧片段，每帧裁剪出 3 个图像块，总计有 30 个视图。
    对于特别大的模型，GPU 显存可能连一个视频都放不下。对于这种情况，您可以在配置文件的 `model['test_cfg']` 中设置 `max_testing_views=n`。
    如此设置，在模型推理过程中，一个批只会使用 n 个视图，以节省显存。

- **如何保存测试结果？**

    测试时，用户可在运行指令中设置可选项 `--out xxx.json/pkl/yaml` 来输出结果文件，以供后续检查。输出的测试结果顺序和测试集顺序保持一致。
    除此之外，MMAction2 也在 [`tools/analysis/eval_metric.py`](/tools/analysis/eval_metric.py) 中提供了分析工具，用于结果文件的模型评估。

## 部署

- **为什么由 MMAction2 转换的 ONNX 模型在转换到其他框架（如 TensorRT）时会抛出错误？**

    目前只能确保 MMAction2 中的模型与 ONNX 兼容。但是，ONNX 中的某些算子可能不受其他框架支持，例如 [这个问题](https://github.com/open-mmlab/mmaction2/issues/414) 中的 TensorRT。当这种情况发生时，如果 `pytorch2onnx.py` 没有出现问题，转换过去的 ONNX 模型也通过了数值检验，可以提 issue 让社区提供帮助。
