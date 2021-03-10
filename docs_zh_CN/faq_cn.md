# 常见问题解答

我们在这里列出了用户们遇到的一些常见问题，及相应的解决方案。
如果您发现了任何社区中经常出现的问题，也有了相应的解决方案，欢迎充实本文档来帮助他人。
如果本文档不包括您的问题，欢迎使用提供的[模板](/.github/ISSUE_TEMPLATE/error-report.md)创建问题，还请确保您在模板中填写了所有必需的信息。

## 安装

- **"No module named 'mmcv.ops'"; "No module named 'mmcv._ext'"**

    1. 使用 `pip uninstall mmcv` 卸载环境中已安装的 `mmcv`。
    2. 遵循[安装文档](https://mmcv.readthedocs.io/en/latest/#installation)以安装 `mmcv-full`。

- **"OSError: MoviePy Error: creation of None failed because of the following error"**

    参照[安装文档](https://github.com/open-mmlab/mmaction2/blob/master/docs/install.md#requirements)
    1. 对于 Windows 用户，[ImageMagick](https://www.imagemagick.org/script/index.php) 不再被 MoviePy 自动检测，
    需要提供名为 `magick` 的 ImageMagick 二进制包的路径，来修改 `moviepy/config_defaults.py` 文件
    2. 对于 Linux 用户，如果 ImageMagick 没有被 moviepy 检测，需要注释掉 `/etc/ImageMagick-6/policy.xml` 文件中的 `<policy domain="path" rights="none" pattern="@*" />`，也即改为 `<!-- <policy domain="path" rights="none" pattern="@*" /> -->`。

## 数据

- **FileNotFound 如 `No such file or directory: xxx/xxx/img_00300.jpg`**

    在 MMAction2 中，我们对帧数据集将 `start_index=1` 设为默认值，而对视频数据集将 `start_index=0` 设为默认值。
    如果用户在数据的第一帧或最后一帧遇到 `文件未找到(FileNotFound)` 的错误，则需要根据每个视频首帧（也即 `xxx_00000.jpg` 或 `xxx_00001.jpg`）的偏移量，对配置文件中的数据处理流水线的 `start_index` 值进行相应的修改。

- **我应该如何处理数据集中传入视频的尺寸？是把所有视频调整为固定尺寸，如 “340x256”，还是把所有视频的短边调整成相同的长度（256像素或320像素）？**

    从基准测试来看，总体来说，后者（把所有视频的短边调整成相同的长度）效果更好，所以我们将“调整尺寸为短边256像素”作为默认的数据处理方式。关于相关的基准测试，你可以在 [TSN 数据基准测试](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/tsn) 和 [SlowOnly 数据基准测试](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/tsn)中查看结果。

- **不匹配的数据流水线项值导致的错误，如 `KeyError: 'total_frames'`**

    对于视频和帧，我们都有相应的流水线来处理。

    **对于视频**，我们应该在处理时首先对它们进行解码。可选的解码方式，有 `DecordInit & DecordDecode`, `OpenCVInit & OpenCVDecode`, `PyAVInit & PyAVDecode` 等等。可以参照[这个例子](https://github.com/open-mmlab/mmaction2/blob/023777cfd26bb175f85d78c455f6869673e0aa09/configs/recognition/slowfast/slowfast_r50_video_4x16x1_256e_kinetics400_rgb.py#L47-L49)。

    **对于帧**，它们已经事先在本地进行了解码，所以使用 `RawFrameDecode` 对帧处理即可。可以参照[这个例子](https://github.com/open-mmlab/mmaction2/blob/023777cfd26bb175f85d78c455f6869673e0aa09/configs/recognition/slowfast/slowfast_r50_8x8x1_256e_kinetics400_rgb.py#L49)。

    `KeyError: 'total_frames'` 是因为错误地使用了 `RawFrameDecode` 来处理视频。当输入是视频的时候，程序是无法事先得到 `total_frame` 的。

## 训练

- **如何使用训练过的识别器作为骨架的预训练模型？**

    参照[使用预训练模型](https://github.com/open-mmlab/mmaction2/blob/master/docs/tutorials/2_finetune.md#use-pre-trained-model)，
    如果想对整个网络使用预训练模型，可以在配置文件中，将 `load_from` 设置为预训练模型的链接。

    如果只想对骨架使用预训练模型，可以在配置文件中，将骨架 `backbone` 中的 `pretrained` 设置为预训练模型的地址或链接。
    在训练时，预训练模型中无法与骨架对应的参数会被忽略。

- **如何实时可视化训练集和验证集的损失曲线？**

    使用 `log_config` 中的 `TensorboardLoggerHook`，如：

    ```python
    log_config=dict(
        interval=20,
        hooks=[
            dict(type='TensorboardLoggerHook')
        ]
    )
    ```

    你可以参照 [tutorials/1_config.md](tutorials/1_config.md)，[tutorials/7_customize_runtime.md](tutorials/7_customize_runtime.md#log-config)，和[这个例子](https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/tsm/tsm_r50_1x1x8_50e_kinetics400_rgb.py#L118)了解更多相关内容。

- **在 batchnorm.py 中抛出错误: Expected more than 1 value per channel when training**

    为使用批正则，批大小应该大于 1，如果构建数据集时， `drop_last` 被设为 `False`，有时一个训练轮次的最后一个批次的批大小可能为 1，于是训练时会抛出错误，你可以将 `drop_last` 设为 `True` 来避免这个错误：

    ```python
    train_dataloader=dict(drop_last=True)
    ```

- **如何在微调模型参数时，冻结骨架中某些阶段的参数？**

    你可以参照 [`def _freeze_stages()`](https://github.com/open-mmlab/mmaction2/blob/0149a0e8c1e0380955db61680c0006626fd008e9/mmaction/models/backbones/x3d.py#L458) 和 [`frozen_stages`](https://github.com/open-mmlab/mmaction2/blob/0149a0e8c1e0380955db61680c0006626fd008e9/mmaction/models/backbones/x3d.py#L183-L184)，记得在分布式训练和测试时，需要设置 `find_unused_parameters = True`。

    实际上，除了少数模型，如 C3D 等，用户都能通过设置 `frozen_stages` 来冻结模型参数，因为大多数骨架继承自 `ResNet` 和 `ResNet3D`，而这两个模型都支持 `_freeze_stages()`。

## 测试

- **如何将预测分值用 softmax 归一化到 [0, 1] 区间内？**

    可以通过设置 `model['test_cfg'] = dict(average_clips='prob')` 来实现。

- **如果模型太大，连一个测试样例都没法放进显存，怎么办？**

    默认情况下，3D 模型是以 `10 clips x 3 crops` 的设置进行测试的，也即采样 10 个帧片段，每帧裁剪出 3 个图像块出来，放入模型的图像数是训练时的 30 倍。
    对于特别大的模型，GPU 显存可能连一个视频都放不下。对于这种情况，你可以在配置文件的 `model['test_cfg']` 中设置 `max_testing_views=n`。
    这么设置后，测试时放入模型的图像数最多时训练时的 n 倍。

- **如何展示测试结果？**

    测试时，我们可以使用指令 `--out xxx.json/pkl/yaml` 来输出结果文件，以供后续检查。输出的测试结果和测试集的顺序保持一致。
    除此之外，我们也在 [`tools/analysis/eval_metric.py`](/tools/analysis/eval_metric.py) 中提供了使用结果文件评估模型的分析工具。

## 部署

- **为什么由 MMAction2 转换的 ONNX 模型在转换到其他框架（如 TensorRT）时会抛出错误？**

    目前，我们只能确保 mmaction2 中的模型与 onnx 兼容。但是，onnx 中的某些算子可能不受其他框架支持，例如[这个问题](https://github.com/open-mmlab/mmaction2/issues/414)中的 TensorRT。当这种情况发生时，如果我们的 `pytorch2onnx.py` 没有出现问题，转换过去的 onnx 模型也通过了数值检验，建议您在对方框架的社区中提问。
