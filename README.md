<div align="center">
  <img src="docs/imgs/mmaction2_logo.png" width="500"/>
</div>

## Introduction

[![Documentation](https://readthedocs.org/projects/mmaction2/badge/?version=latest)](https://mmaction2.readthedocs.io/en/latest/)
[![actions](https://github.com/open-mmlab/mmaction2/workflows/build/badge.svg)](https://github.com/open-mmlab/mmaction2/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmaction2/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmaction2)
[![PyPI](https://img.shields.io/pypi/v/mmaction2)](https://pypi.org/project/mmaction2/)
[![LICENSE](https://img.shields.io/github/license/open-mmlab/mmaction2.svg)](https://github.com/open-mmlab/mmaction2/blob/master/LICENSE)
[![Average time to resolve an issue](https://isitmaintained.com/badge/resolution/open-mmlab/mmaction2.svg)](https://github.com/open-mmlab/mmaction2/issues)
[![Percentage of issues still open](https://isitmaintained.com/badge/open/open-mmlab/mmaction2.svg)](https://github.com/open-mmlab/mmaction2/issues)


MMAction2 is an open-source toolbox for action understanding based on PyTorch.
It is a part of the [OpenMMLab](http://openmmlab.org/) project.

The master branch works with **PyTorch 1.3+**.

<div align="center">
  <img src="docs/imgs/mmaction2_overview.gif" width="600px"/>
</div>

### Major Features

- **Modular design**

  We decompose the action understanding framework into different components and one can easily construct a customized
  action understanding framework by combining different modules.

- **Support for various datasets**

  The toolbox directly supports multiple datasets, UCF101, Kinetics-400, Something-Something V1&V2, Moments in Time, Multi-Moments in Time, THUMOS14, etc.

- **Support for multiple action understanding frameworks**

  MMAction2 implements popular frameworks for action understanding:

  - For action recognition, various algorithms are implemented, including TSN, TSM, TIN, R(2+1)D, I3D, SlowOnly, SlowFast, CSN, Non-local, etc.

  - For temporal action localization, we implement BSN, BMN, SSN.

- **Well tested and documented**

  We provide detailed documentation and API reference, as well as unittests.

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Changelog

v0.6.0 was released in 2/9/2020. Please refer to [changelog.md](docs/changelog.md) for details and release history.

## Benchmark
| Model  |input| io backend | batch size x gpus | MMAction2 (s/iter) | MMAction (s/iter) | Temporal-Shift-Module (s/iter) | PySlowFast (s/iter) |
| :--- | :---------------:|:---------------:| :---------------:| :---------------:  | :--------------------: | :----------------------------: | :-----------------: |
| [TSN](/configs/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb.py)| 256p rawframes |Memcached| 32x8|**[0.32](https://download.openmmlab.com/mmaction/benchmark/recognition/mmaction2/tsn_256p_rawframes_memcahed_32x8.zip)** | [0.38](https://download.openmmlab.com/mmaction/benchmark/recognition/mmaction/tsn_256p_rawframes_memcached_32x8.zip)| [0.42](https://download.openmmlab.com/mmaction/benchmark/recognition/temporal_shift_module/tsn_256p_rawframes_memcached_32x8.zip)| x |
| [TSN](/configs/recognition/tsn/tsn_r50_video_1x1x3_100e_kinetics400_rgb.py)| 256p dense-encoded video |Disk| 32x8|**[0.61](https://download.openmmlab.com/mmaction/benchmark/recognition/mmaction2/tsn_256p_fast_videos_disk_32x8.zip)**| x | x | TODO |
|[I3D heavy](/configs/recognition/i3d/i3d_r50_video_heavy_8x8x1_100e_kinetics400_rgb.py)|256p videos|Disk |8x8| **[0.34](https://download.openmmlab.com/mmaction/benchmark/recognition/mmaction2/i3d_heavy_256p_videos_disk_8x8.zip)** | x | x | [0.44](https://download.openmmlab.com/mmaction/benchmark/recognition/pyslowfast/pysf_i3d_r50_8x8_video.log) |
| [I3D](/configs/recognition/i3d/i3d_r50_32x2x1_100e_kinetics400_rgb.py)|256p rawframes|Memcached|8x8| **[0.43](https://download.openmmlab.com/mmaction/benchmark/recognition/mmaction2/i3d_256p_rawframes_memcahed_8x8.zip)** | [0.56](https://download.openmmlab.com/mmaction/benchmark/recognition/mmaction/i3d_256p_rawframes_memcached_8x8.zip) | x | x |
| [TSM](/configs/recognition/tsm/tsm_r50_1x1x8_50e_kinetics400_rgb.py) |256p rawframes|Memcached| 8x8|**[0.31](https://download.openmmlab.com/mmaction/benchmark/recognition/mmaction2/tsm_256p_rawframes_memcahed_8x8.zip)** | x | [0.41](https://download.openmmlab.com/mmaction/benchmark/recognition/temporal_shift_module/tsm_256p_rawframes_memcached_8x8.zip) | x |
| [Slowonly](/configs/recognition/slowonly/slowonly_r50_video_4x16x1_256e_kinetics400_rgb.py)|256p videos|Disk|8x8 | **[0.32](https://download.openmmlab.com/mmaction/benchmark/recognition/mmaction2/slowonly_256p_videos_disk_8x8.zip)** | TODO | x | [0.34](https://download.openmmlab.com/mmaction/benchmark/recognition/pyslowfast/pysf_slowonly_r50_4x16_video.log) |
| [Slowfast](/configs/recognition/slowfast/slowfast_r50_video_4x16x1_256e_kinetics400_rgb.py)|256p videos|Disk|8x8 | **[0.69](https://download.openmmlab.com/mmaction/benchmark/recognition/mmaction2/slowfast_256p_videos_disk_8x8.zip)** | x | x | [1.04](https://download.openmmlab.com/mmaction/benchmark/recognition/pyslowfast/pysf_slowfast_r50_4x16_video.log) |
| [R(2+1)D](/configs/recognition/r2plus1d/r2plus1d_r34_video_8x8x1_180e_kinetics400_rgb.py)|256p videos |Disk| 8x8|**[0.45](https://download.openmmlab.com/mmaction/benchmark/recognition/mmaction2/r2plus1d_256p_videos_disk_8x8.zip)** | x | x | x |

Details can be found in [benchmark](docs/benchmark.md).

## ModelZoo
Supported methods for action recognition:
- [x] [TSN](configs/recognition/tsn/README.md)
- [x] [TSM](configs/recognition/tsm/README.md)
- [x] [R(2+1)D](configs/recognition/r2plus1d/README.md)
- [x] [I3D](configs/recognition/i3d/README.md)
- [x] [SlowOnly](configs/recognition/slowonly/README.md)
- [x] [SlowFast](configs/recognition/slowfast/README.md)
- [x] [CSN](configs/recognition/csn/README.md)
- [x] [TIN](configs/recognition/tin/README.md)
- [x] [TPN](configs/recognition/tpn/README.md)

Supported methods for action localization:
- [x] [BMN](configs/localization/bmn/README.md)
- [x] [BSN](configs/localization/bsn/README.md)

Results and models are available in the *README.md* of each method's config directory.
A summary can be found in the [**model zoo**](https://mmaction2.readthedocs.io/en/latest/modelzoo.html) page.

## Installation

Please refer to [install.md](docs/install.md) for installation.

## Data Preparation

Please refer to [data_preparation.md](docs/data_preparation.md) for a general knowledge of data preparation.

## Get Started

Please see [getting_started.md](docs/getting_started.md) for the basic usage of MMAction2.
There are also tutorials for [finetuning models](docs/tutorials/finetune.md), [adding new dataset](docs/tutorials/new_dataset.md), [designing data pipeline](docs/tutorials/data_pipeline.md), [exporting model to onnx](docs/tutorials/export_model.md) and [adding new modules](docs/tutorials/new_modules.md).

A Colab tutorial is also provided. You may preview the notebook [here](demo/mmaction2_tutorial.ipynb) or directly [run](https://colab.research.google.com/github/open-mmlab/mmaction2/blob/master/demo/mmaction2_tutorial.ipynb) on Colab.

## Contributing

We appreciate all contributions to improve MMAction2. Please refer to [CONTRIBUTING.md](.github/CONTRIBUTING.md) for the contributing guideline.

## Acknowledgement

MMAction2 is an open source project that is contributed by researchers and engineers from various colleges and companies.
We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks.
We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their own new models.
