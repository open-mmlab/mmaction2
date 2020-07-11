<div align="center">
  <img src="docs/imgs/mmaction2-logo.png" width="500"/>
</div>

<div align="left">
	<a href='https://mmaction2.readthedocs.io/en/latest/?badge=latest'>
	    <img src='https://readthedocs.org/projects/mmaction2/badge/?version=latest' alt='Documentation Status' />
	</a>
  <a href="https://codecov.io/gh/open-mmlab/mmaction2">
      <img src="https://codecov.io/gh/open-mmlab/mmaction2/branch/master/graph/badge.svg" />
  </a>
  <a href="https://github.com/open-mmlab/mmaction2/blob/master/LICENSE">
	    <img src="https://img.shields.io/github/license/open-mmlab/mmaction2.svg">
	</a>
</div>

Documentation: https://mmaction2.readthedocs.io/.

## Introduction

MMAction2 is an open-source toolbox for action understanding based on PyTorch.
It is a part of the [OpenMMLab](http://openmmlab.org/) project.

The master branch works with **PyTorch 1.3+**.

<div align="center">
  <img src="demo/demo.gif" width="600px"/>
</div>

### Major Features

- **Modular design**

  We decompose the action understanding framework into different components and one can easily construct a customized
  action understanding framework by combining different modules.

- **Support for various datasets**

  The toolbox directly supports multiple datasets, UCF101, Kinetics-400, Something-Something V1&V2, Moments in Time, Multi-Moments in Time, THUMOS14, etc.

- **Support for multiple action understanding frameworks**

  MMAction2 implements popular frameworks for action understanding:

  - For action recognition, various algorithms are implemented, including TSN, TSM, R(2+1)D, I3D, SlowOnly, SlowFast.

  - For temporal action localization, we implement BSN, BMN.

- **Well tested and documented**

  We provide detailed documentation and API reference, as well as unittests.

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Benchmark and Model Zoo

MMAction2 supports various models and is more efficient in training.
We compare with other popular codebases and the results are shown as below.

| Model | MMAction2 (s/iter) | MMAction (s/iter) | Temporal-Shift-Module (s/iter) | PySlowFast (s/iter) |
| :--- | :---------------: | :--------------------: | :----------------------------: | :-----------------: |
| [TSN](/configs/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb.py) | **0.29** | 0.36 | 0.45 | x |
| [I3D (setting1)](/configs/recognition/i3d/i3d_r50_32x2x1_100e_kinetics400_rgb.py) | **0.45** | 0.58 | x | x |
| [I3D (setting2)](/configs/recognition/i3d/i3d_r50_8x8x1_100e_kinetics400_rgb.py) | **0.32** | x | x | 0.56 |
| [TSM](/configs/recognition/tsm/tsm_r50_1x1x8_50e_kinetics400_rgb.py) | **0.30** | x | 0.38 | x |
| [Slowonly](/configs/recognition/slowonly/slowonly_r50_4x16x1_256e_kinetics400_rgb.py) | **0.30** | x | x | 1.03 |
| [Slowfast](/configs/recognition/slowfast/slowfast_r50_4x16x1_256e_kinetics400_rgb.py) | **0.80** | x | x | 1.40 |
| [R(2+1)D](/configs/recognition/r2plus1d/r2plus1d_r34_8x8x1_180e_kinetics400_rgb.py) | **0.48** | x | x | x |

Supported methods for action recognition:
- [x] [TSN](configs/recognition/tsn/README.md)
- [x] [TSM](configs/recognition/tsm/README.md)
- [x] [R(2+1)D](configs/recognition/r2plus1d/README.md)
- [x] [I3D](configs/recognition/i3d/README.md)
- [x] [SlowOnly](configs/recognition/slowonly/README.md)
- [x] [SlowFast](configs/recognition/slowfast/README.md)

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
There are also tutorials for [finetuning models](docs/tutorials/finetune.md), [adding new dataset](docs/tutorials/new_dataset.md), [designing data pipeline](docs/tutorials/data_pipeline.md), and [adding new modules](docs/tutorials/new_modules.md).

## Contributing

We appreciate all contributions to improve MMAction2. Please refer to [CONTRIBUTING.md](.github/CONTRIBUTING.md) for the contributing guideline.

## Acknowledgement

MMAction2 is an open source project that is contributed by researchers and engineers from various colleges and companies.
We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks.
We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their own new models.
