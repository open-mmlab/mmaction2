# FAQ

We list some common issues faced by many users and their corresponding solutions here.
Feel free to enrich the list if you find any frequent issues and have ways to help others to solve them.
If the contents here do not cover your issue, please create an issue using the [provided templates](/.github/ISSUE_TEMPLATE/error-report.md) and make sure you fill in all required information in the template.


## Installation

- **"No module named 'mmcv.ops'"; "No module named 'mmcv._ext'"**

    1. Uninstall existing mmcv in the environment using `pip uninstall mmcv`.
    2. Install mmcv-full following the [installation instruction](https://mmcv.readthedocs.io/en/latest/#installation).

- **"OSError: MoviePy Error: creation of None failed because of the following error"**

    Refer to [install.md](https://github.com/open-mmlab/mmaction2/blob/master/docs/install.md#requirements)
    1. For Windows users, [ImageMagick](https://www.imagemagick.org/script/index.php) will not be automatically detected by MoviePy,
    there is a need to modify `moviepy/config_defaults.py` file by providing the path to the ImageMagick binary called `magick`,
    like `IMAGEMAGICK_BINARY = "C:\\Program Files\\ImageMagick_VERSION\\magick.exe"`
    2. For Linux users, there is a need to modify the `/etc/ImageMagick-6/policy.xml` file by commenting out
    `<policy domain="path" rights="none" pattern="@*" />` to `<!-- <policy domain="path" rights="none" pattern="@*" /> -->`,
    if ImageMagick is not detected by moviepy.

## Data

- **FileNotFound like `No such file or directory: xxx/xxx/img_00300.jpg`**

    In our repo, we set `start_index=1` as default value for rawframe dataset, and `start_index=0` as default value for video dataset.
    If users encounter FileNotFound error for the first or last frame of the data, there is a need to check the files begin with offset 0 or 1,
    that is `xxx_00000.jpg` or `xxx_00001.jpg`, and then change the `start_index` value of data pipeline in configs.


## Training

- **How to just use trained recognizer models for backbone pre-training ?**

    Refer to [Use Pre-Trained Model](https://github.com/open-mmlab/mmaction2/blob/master/docs/tutorials/finetune.md#use-pre-trained-model),
    in order to use the pre-trained model for the whole network, the new config adds the link of pre-trained models in the `load_from`.

    And to use backbone for pre-training, you can change `pretrained` value in the backbone dict of config files to the checkpoint path / url.
    When training, the unexpected keys will be ignored.


## Testing

- **How to make predicted score normalized by softmax within [0, 1] ?**

    change this in the config, make `test_cfg = dict(average_clips='prob')`.

## Deploying

- **Why is the onnx model converted by mmaction2 throwing error when converting to other frameworks such as TensorRT?**

    For now, we can only make sure that models in mmaction2 are onnx-compatible. However, some operations in onnx may be unsupported by your target framework for deployment, e.g. TensorRT in [this issue](https://github.com/open-mmlab/mmaction2/issues/414). When such situation occurs, we suggest you raise an issue in the repo of your target framework as long as `pytorch2onnx.py` works well and is verified numerically.
