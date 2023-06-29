# 常见问题解答

## 概述

我们在这里列出了许多用户常遇到的问题以及相应的解决方案。

- [常见问题解答](#常见问题解答)
  - [概述](#概述)
  - [安装](#安装)
  - [数据](#数据)
  - [训练](#训练)
  - [测试](#测试)

如果您发现任何频繁出现的问题并且有解决方法，欢迎在列表中补充。如果这里的内容没有涵盖您的问题，请使用[提供的模板](https://github.com/open-mmlab/mmaction2/tree/main/.github/ISSUE_TEMPLATE/error-report.md)创建一个问题，并确保在模板中填写所有必要的信息。

## 安装

- **"No module named 'mmcv.ops'"; "No module named 'mmcv.\_ext'"**

  1. 使用 `pip uninstall mmcv` 命令卸载环境中的现有 mmcv。
  2. 参照[安装说明](https://mmcv.readthedocs.io/en/2.x/get_started/installation.html#install-mmcv)安装 mmcv。

- **"OSError: MoviePy Error: creation of None failed because of the following error"**

  使用 `pip install moviepy` 安装。更多信息可以参考[官方安装文档](https://zulko.github.io/moviepy/install.html), 请注意（根据这个 [issue](https://github.com/Zulko/moviepy/issues/693)）：

  1. 对于 Windows 用户，[ImageMagick](https://www.imagemagick.org/script/index.php) 不会自动被 MoviePy 检测到，需要修改 `moviepy/config_defaults.py` 文件，提供 ImageMagick 二进制文件 `magick` 的路径，例如 `IMAGEMAGICK_BINARY = "C:\\Program Files\\ImageMagick_VERSION\\magick.exe"`
  2. 对于 Linux 用户，如果 MoviePy 没有检测到 ImageMagick，需要修改 `/etc/ImageMagick-6/policy.xml` 文件，将 `<policy domain="path" rights="none" pattern="@*" />` 注释掉，改为 `<!-- <policy domain="path" rights="none" pattern="@*" /> -->`。

- **"即使我已经安装了 XXCODEBASE，为什么还会收到 'Please install XXCODEBASE to use XXX' 的错误消息?"**

  您收到该错误消息是因为我们的项目无法从 XXCODEBASE 中导入一个函数或类。您可以尝试运行相应的代码行来查看发生了什么。一个可能的原因是，在 OpenMMLAB 的某些代码库中，您需要在安装它们之前先安装 mmcv 和 mmengine。您可以按照[教程](https://mmaction2.readthedocs.io/en/latest/get_started/installation.html#installation)来安装它们。

## 数据

- **FileNotFound 错误，例如 `No such file or directory: xxx/xxx/img_00300.jpg`**

  在我们的仓库中，我们将 `start_index=1` 设置为 rawframe 数据集的默认值，将 `start_index=0` 设置为视频数据集的默认值。如果用户遇到数据的第一帧或最后一帧的 FileNotFound 错误，需要检查以 0 或 1 作为偏移量开始的文件，例如 `xxx_00000.jpg` 或 `xxx_00001.jpg`，然后在配置文件中更改数据处理流水线的 `start_index` 值。

- **我们应该如何预处理数据集中的视频？将它们调整为固定大小（所有视频的高宽比相同），例如 `340x256`（1），还是调整它们使得所有视频的短边具有相同的长度（256px 或 320px）（2）？**

  我们尝试过这两种预处理方法，并发现（2）通常是更好的解决方案，因此我们使用（2）作为默认的预处理设置，短边长度为 256px。我们对这些预处理方法进行了基准测试，您可以在[TSN 数据基准测试](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/tsn)和[SlowOnly 数据基准测试](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/slowonly)中找到结果。

- **数据处理流水线中的项不匹配导致出现类似 `KeyError: 'total_frames'` 的错误**

  我们有用于处理视频和帧的两个处理流水线。

  **对于视频**，我们应该在处理流水线中动态解码视频，所以在这种情况下应该使用 `DecordInit & DecordDecode`、`OpenCVInit & OpenCVDecode` 或 `PyAVInit & PyAVDecode` 这样的配对，例如[这个示例](https://github.com/open-mmlab/mmaction2/blob/main/configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py#L14-L16)。

  **对于帧**，图像已经在离线状态下解码，所以在这种情况下应该使用 `RawFrameDecode` 这样的处理流水线项，例如[这个示例](https://github.com/open-mmlab/mmaction2/blob/main/configs/recognition/trn/trn_imagenet-pretrained-r50_8xb16-1x1x8-50e_sthv1-rgb.py#L17)。

  `KeyError: 'total_frames'` 是由于错误地将 `RawFrameDecode` 步骤用于视频，因为当输入是视频时，无法预先获取 `total_frames`。

## 训练

- **如何只使用训练好的识别模型进行主干网络的预训练？**

  为了使用预训练模型进行整个网络的训练，新的配置文件在 `load_from` 中添加了预训练模型的链接。

  要使用主干进行预训练，可以将配置文件中主干部分的 `pretrained` 值更改为权重路径/URL。在训练时，未预料到的键将被忽略。

- **在微调模型时如何固定主干的某些阶段？**

  您可以参考 [`def _freeze_stages()`](https://github.com/open-mmlab/mmaction2/blob/main/mmaction/models/backbones/resnet3d.py#L791) 和 [`frozen_stages`](https://github.com/open-mmlab/mmaction2/blob/main/mmaction/models/backbones/resnet3d.py#L369-L370)。
  提醒在配置文件中设置 `find_unused_parameters = True`，以进行分布式训练或测试。

  实际上，除了少数模型，如 C3D 等，用户可以设置 `frozen_stages` 来冻结主干的阶段，因为几乎所有继承自 `ResNet` 和 `ResNet3D` 的主干都支持内部函数 `_freeze_stages()`。

- **如何在配置文件中设置 memcached ？**

  在 MMAction2 中，您可以将 memcached 的参数传递给用于视频数据集的 `class DecordInit` 或用于原始帧数据集的 `RawFrameDecode`。有关更多细节，请参阅 MMEngine 中的 [`class FileClient`](https://github.com/open-mmlab/mmaction2/blob/main/mmaction/data/pipelines/file_client.py)。以下是一个示例，演示如何在原始帧数据集中使用 memcached：

  ```python
  mc_cfg = dict(server_list_cfg='server_list_cfg', client_cfg='client_cfg', sys_path='sys_path')

  train_pipeline = [
    ...
    dict(type='RawFrameDecode', io_backend='memcached', **mc_cfg),
    ...
  ]
  ```

- **如何在配置文件中设置 `load_from` 的值以微调模型？**

  在 MMAction2 中，我们将 `load_from=None` 设置为 `configs/_base_/default_runtime.py` 中的默认值，并且由于[继承设计](https://github.com/open-mmlab/mmaction2/tree/main/docs/en/user_guides/config.md)，用户可以直接通过在其配置文件中设置 `load_from` 来更改它。

- **如何在训练时使用 `RawFrameDataset`？**

  在 MMAction2 1.x 版本中，大多数配置文件默认使用 `VideoDataset` 作为数据集类型，这对于文件存储更加友好。如果您想使用 `RawFrameDataset`，需要进行两个修改步骤：

  - `dataset` 相关：
    将 `train_dataloader`/`val_dataloader`/`test_dataloader` 中的 `dataset` 从

    ```
    dataset=dict(
        type=VideoDataset,
        data_prefix=dict(video=xxx),
        ...)
    ```

    修改为

    ```
    dataset=dict(
        type=RawFrameDataset,
        data_prefix=dict(img=xxx),
        filename_tmpl='{:05}.jpg',
        ...)
    ```

    数据集的其他字段不需要修改。请确保 `filename_tmpl` 与帧数据匹配，并参考[配置文件文档](../user_guides/config.md)了解更多关于配置文件的详细信息。

  - `transform` 相关：在 `train_pipeline`/`val_pipeline`/`test_pipeline` 中删除 `dict(type='DecordInit', **file_client_args)`，将 `dict(type='DecordDecode')` 修改为 `dict(type='RawFrameDecode', **file_client_args)`，并确保在配置文件中定义了 `file_client_args = dict(io_backend='disk')`。

  有关自定义数据集的更多修改，请参考[准备数据集](../user_guides/prepare_dataset.md)和[自定义数据集](../advanced_guides/customize_dataset.md)。

## 测试

- **如何使预测得分在 softmax 内归一化到 \[0, 1\] ?**

  在配置文件中将 `model.cls_head.average_clips` 设置为 `'prob'`。

- **如果模型过大，GPU 内存无法容纳甚至只有一个测试样本怎么办？**

  默认情况下，3D 模型使用 10 个 clips x 3 个 crops 进行测试，总共有 30 个视图。对于非常大的模型，即使只有一个测试样本，GPU 内存也无法容纳（因为有 30 个视图）。为了解决这个问题，您可以在配置文件的 `model['test_cfg']` 中设置 `max_testing_views=n`。这样，在前向传播过程中，会使用 n 个视图作为一个批次，以节省 GPU 内存的使用。
