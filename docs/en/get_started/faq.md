# FAQ

## Outline

We list some common issues faced by many users and their corresponding solutions here.

- [FAQ](#faq)
  - [Outline](#outline)
  - [Installation](#installation)
  - [Data](#data)
  - [Training](#training)
  - [Testing](#testing)

Feel free to enrich the list if you find any frequent issues and have ways to help others to solve them.
If the contents here do not cover your issue, please create an issue using the [provided templates](https://github.com/open-mmlab/mmaction2/tree/main/.github/ISSUE_TEMPLATE/error-report.md) and make sure to fill in all required information in the template.

## Installation

- **"No module named 'mmcv.ops'"; "No module named 'mmcv.\_ext'"**

  1. Uninstall existing mmcv in the environment using `pip uninstall mmcv`
  2. Install mmcv following the [installation instruction](https://mmcv.readthedocs.io/en/2.x/get_started/installation.html#install-mmcv)

- **"OSError: MoviePy Error: creation of None failed because of the following error"**

  Refer to [install.md](https://github.com/open-mmlab/mmaction2/blob/master/docs/install.md#requirements)

  1. For Windows users, [ImageMagick](https://www.imagemagick.org/script/index.php) will not be automatically detected by MoviePy, there is a need to modify `moviepy/config_defaults.py` file by providing the path to the ImageMagick binary called `magick`, like `IMAGEMAGICK_BINARY = "C:\\Program Files\\ImageMagick_VERSION\\magick.exe"`
  2. For Linux users, there is a need to modify the `/etc/ImageMagick-6/policy.xml` file by commenting out `<policy domain="path" rights="none" pattern="@*" />` to `<!-- <policy domain="path" rights="none" pattern="@*" /> -->`, if ImageMagick is not detected by moviepy.

- **"Why I got the error message 'Please install XXCODEBASE to use XXX' even if I have already installed XXCODEBASE?"**

  You got that error message because our project failed to import a function or a class from XXCODEBASE. You can try to run the corresponding line to see what happens. One possible reason is, for some codebases in OpenMMLAB, you need to install mmcv and mmengine before you install them. You could follow this [tutorial](https://mmaction2.readthedocs.io/en/latest/get_started/installation.html#installation) to install them.

## Data

- **FileNotFound like `No such file or directory: xxx/xxx/img_00300.jpg`**

  In our repo, we set `start_index=1` as default value for rawframe dataset, and `start_index=0` as default value for video dataset.
  If users encounter FileNotFound error for the first or last frame of the data, there is a need to check the files begin with offset 0 or 1,
  that is `xxx_00000.jpg` or `xxx_00001.jpg`, and then change the `start_index` value of data pipeline in configs.

- **How should we preprocess the videos in the dataset? Resizing them to a fix size(all videos with the same height-width ratio) like `340x256` (1) or resizing them so that the short edges of all videos are of the same length (256px or 320px) (2)**

  We have tried both preprocessing approaches and found (2) is a better solution in general, so we use (2) with short edge length 256px as the default preprocessing setting. We benchmarked these preprocessing approaches and you may find the results in [TSN Data Benchmark](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/tsn) and [SlowOnly Data Benchmark](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/slowonly).

- **Mismatched data pipeline items lead to errors like `KeyError: 'total_frames'`**

  We have both pipeline for processing videos and frames.

  **For videos**, We should decode them on the fly in the pipeline, so pairs like `DecordInit & DecordDecode`, `OpenCVInit & OpenCVDecode`, `PyAVInit & PyAVDecode` should be used for this case like [this example](https://github.com/open-mmlab/mmaction2/blob/main/configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py#L14-L16).

  **For Frames**, the image has been decoded offline, so pipeline item `RawFrameDecode` should be used for this case like [this example](https://github.com/open-mmlab/mmaction2/blob/main/configs/recognition/trn/trn_imagenet-pretrained-r50_8xb16-1x1x8-50e_sthv1-rgb.py#L17).

  `KeyError: 'total_frames'` is caused by incorrectly using `RawFrameDecode` step for videos, since when the input is a video, it can not get the `total_frames` beforehand.

## Training

- **How to just use trained recognizer models for backbone pre-training?**

  In order to use the pre-trained model for the whole network, the new config adds the link of pre-trained models in the `load_from`.

  And to use backbone for pre-training, you can change `pretrained` value in the backbone dict of config files to the checkpoint path / url.
  When training, the unexpected keys will be ignored.

- **How to fix stages of backbone when finetuning a model?**

  You can refer to [`def _freeze_stages()`](https://github.com/open-mmlab/mmaction2/blob/main/mmaction/models/backbones/resnet3d.py#L791) and [`frozen_stages`](https://github.com/open-mmlab/mmaction2/blob/main/mmaction/models/backbones/resnet3d.py#L369-L370).
  Reminding to set `find_unused_parameters = True` in config files for distributed training or testing.

  Actually, users can set `frozen_stages` to freeze stages in backbones except C3D model, since almost all backbones inheriting from `ResNet` and `ResNet3D` support the inner function `_freeze_stages()`.

- **How to set memcached setting in config files?**

  In MMAction2, you can pass memcached kwargs to `class DecordInit` for video dataset or `RawFrameDecode` for rawframes dataset.
  For more details, you can refer to \[`class FileClient`\] in MMEngine for more details.
  Here is an example to use memcached for rawframes dataset:

  ```python
  mc_cfg = dict(server_list_cfg='server_list_cfg', client_cfg='client_cfg', sys_path='sys_path')

  train_pipeline = [
    ...
    dict(type='RawFrameDecode', io_backend='memcached', **mc_cfg),
    ...
  ]
  ```

- **How to set `load_from` value in config files to finetune models?**

  In MMAction2, We set `load_from=None` as default in `configs/_base_/default_runtime.py` and owing to [inheritance design](https://github.com/open-mmlab/mmaction2/tree/main/docs/en/user_guides/config.md),
  users can directly change it by setting `load_from` in their configs.

- **How to use `RawFrameDataset` for training?**

  In MMAction2 1.x version, most of the configs take `VideoDataset` as the default dataset type, which is much more friendly to file storage. If you want to use `RawFrameDataset` instead, there are two steps to modify:

  - Dataset:
    modify dataset in `train_dataloader`/`val_dataloader`/`test_dataloader` from

    ```
    dataset=dict(
        type=VideoDataset,
        data_prefix=dict(video=xxx),
        ...)
    ```

    to

    ```
    dataset=dict(
        type=RawFrameDataset,
        data_prefix=dict(img=xxx),
        filename_tmpl='{:05}.jpg',
        ...)
    ```

    remaining fields of `dataset` don't need to be modified. Please make sure that `filename_tmpl` is matching with your frame data, and you can refer to [config document](../user_guides/config.md) for more details about config file.

  - Transforms: delete `dict(type='DecordInit', **file_client_args)`, modify `dict(type='DecordDecode')` to `dict(type='RawFrameDecode', **file_client_args)` in `train_pipeline`/`val_pipeline`/`test_pipeline`, and please make sure that `file_client_args = dict(io_backend='disk')` has been defined in your config.

  For more modifications about customizing datasets, please refer to [prepare dataset](../user_guides/prepare_dataset.md) and [customize dataset](../advanced_guides/customize_dataset.md).

## Testing

- **How to make predicted score normalized by softmax within \[0, 1\]?**

  change this in the config, make `model.cls_head.average_clips = 'prob'`.

- **What if the model is too large and the GPU memory can not fit even only one testing sample?**

  By default, the 3d models are tested with 10clips x 3crops, which are 30 views in total. For extremely large models, the GPU memory can not fit even only one testing sample (cuz there are 30 views). To handle this, you can set `max_testing_views=n` in `model['test_cfg']` of the config file. If so, n views will be used as a batch during forwarding to save GPU memory used.
