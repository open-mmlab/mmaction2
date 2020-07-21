<div align="center">
  <img src="docs/imgs/mmaction2-logo.png" width="500"/>
</div>

<div align="left">
    <a href='https://pypi.org/project/mmaction2/'>
        <img alt="PyPI" src="https://img.shields.io/pypi/v/mmaction2">
    </a>
    <a href='https://mmaction2.readthedocs.io/en/latest/'>
        <img src='https://readthedocs.org/projects/mmaction2/badge/?version=latest' alt='Documentation Status' />
    </a>
    <a href='https://github.com/open-mmlab/mmaction2/actions'>
        <img src='https://github.com/open-mmlab/mmaction2/workflows/build/badge.svg' />
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

## Model Zoo


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

A Colab tutorial is also provided. You may preview the notebook [here](demo/mmaction2_tutorial.ipynb) or directly [run](https://colab.research.google.com/github/open-mmlab/mmaction2/blob/master/demo/mmaction2_tutorial.ipynb) on Colab.

## Contributing

We appreciate all contributions to improve MMAction2. Please refer to [CONTRIBUTING.md](.github/CONTRIBUTING.md) for the contributing guideline.

## Acknowledgement

MMAction2 is an open source project that is contributed by researchers and engineers from various colleges and companies.
We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks.
We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their own new models.

## Contact
If you have any question, please file an issue or contact the authors:

* [<img src="https://github.com/dreamerlin.png" width="24" height="24" alt="@dreamerlin"> @dreamerlin](https://github.com/dreamerlin) Jintao Lin
* [<img src="https://github.com/JoannaLXY.png" width="24" height="24" alt="@JoannaLXY"> @JoannaLXY](https://github.com/JoannaLXY) Xuanyi Li
* [<img src="https://github.com/SuX97.png" width="24" height="24" alt="@SuX97"> @SuX97](https://github.com/SuX97) Su Xu
* [<img src="https://github.com/kennymckormick.png" width="24" height="24" alt="@kennymckormick"> @kennymckormick](https://github.com/kennymckormick) Haodong Duan
