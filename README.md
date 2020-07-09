<div align="center">
  <img src="docs/imgs/mmaction-logo.png" width="500"/>
</div>

## Introduction

<div align="left">
	<a href="http://open-mmlab.pages.gitlab.sz.sensetime.com/mmaction-lite/">
		<img src="https://img.shields.io/badge/docs-latest-blue"/>
	</a>
	<a href="https://codecov.io/gh/open-mmlab/mmaction">
		<img src="https://codecov.io/gh/open-mmlab/mmaction/branch/master/graph/badge.svg"/>
	</a>
	<a href="https://pypi.org/project/mmaction">
		<img src="https://img.shields.io/pypi/pyversions/mmaction.svg?style=plastic"/>
	</a>
	<a href="https://pypi.org/project/mmaction">
		<img src="https://img.shields.io/pypi/v/mmaction"/>
	</a>
	<a href="https://github.com/open-mmlab/mmaction/blob/master/LICENSE">
	    <img src="https://img.shields.io/github/license/open-mmlab/mmaction.svg">
	</a>
</div>

The master branch works with **PyTorch 1.3+**.

MMAction is an open-source toolbox for action understanding based on PyTorch.
It is a part of the [OpenMMLab project](https://github.com/open-mmlab) developed by [Multimedia Laboratory, CUHK](http://mmlab.ie.cuhk.edu.hk/).

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

  MMAction implements popular frameworks for action understanding:

  - For action recognition, various algorithms are implemented, including TSN, TSM, R(2+1)D, I3D, SlowOnly, SlowFast.

  - For temporal action localization, we implement BSN, BMN.

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Benchmark and Model Zoo

Benchmark with other repos are available on [benchmark.md](docs/benchmark.md).

Results and models are available in the **README.md** of each method's config directory.

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

## Installation

Please refer to [install.md](docs/install.md) for installation.

## Data Preparation

Please refer to [data_preparation.md](docs/data_preparation.md) for a general knowledge of data preparation.

## Get Started

Please see [getting_started.md](docs/getting_started.md) for the basic usage of MMAction.

## Contributing

We appreciate all contributions to improve MMAction. Please refer to [CONTRIBUTING.md in MMDetection](https://github.com/open-mmlab/mmdetection/blob/master/.github/CONTRIBUTING.md) for the contributing guideline.

## Acknowledgement

MMAction is an open source project that is contributed by researchers and engineers from various colleges and companies.
We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks.
We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their own new models.

## Citation

If you use this toolbox or benchmark in your research, please cite this project.

```
TODO
```

## Contact

TODO

This repo is currently maintained by xxx ([@xxx](http://github.com/xxx)), ....
