# X3D

[X3D: Expanding Architectures for Efficient Video Recognition](https://openaccess.thecvf.com/content_CVPR_2020/html/Feichtenhofer_X3D_Expanding_Architectures_for_Efficient_Video_Recognition_CVPR_2020_paper.html)

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

This paper presents X3D, a family of efficient video networks that progressively expand a tiny 2D image classification architecture along multiple network axes, in space, time, width and depth. Inspired by feature selection methods in machine learning, a simple stepwise network expansion approach is employed that expands a single axis in each step, such that good accuracy to complexity trade-off is achieved. To expand X3D to a specific target complexity, we perform progressive forward expansion followed by backward contraction. X3D achieves state-of-the-art performance while requiring 4.8x and 5.5x fewer multiply-adds and parameters for similar accuracy as previous work. Our most surprising finding is that networks with high spatiotemporal resolution can perform well, while being extremely light in terms of network width and parameters. We report competitive accuracy at unprecedented efficiency on video classification and detection benchmarks.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/34324155/143019391-6711febb-9e5d-4bec-85b9-65f5179e93a2.png" width="800"/>
</div>

## Results and Models

### Kinetics-400

| frame sampling strategy | resolution | backbone | top1 10-view | top1 30-view |           reference top1 10-view           |           reference top1 30-view           |           config           |           ckpt            |
| :---------------------: | :--------: | :------: | :----------: | :----------: | :----------------------------------------: | :----------------------------------------: | :------------------------: | :-----------------------: |
|         13x6x1          |  160x160   |  X3D_S   |     73.2     |     73.3     | 73.1 \[[SlowFast](https://github.com/facebookresearch/SlowFast/blob/master/MODEL_ZOO.md)\] | 73.5 \[[SlowFast](https://github.com/facebookresearch/SlowFast/blob/master/MODEL_ZOO.md)\] | [config](/configs/recognition/x3d/x3d_s_13x6x1_facebook-kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/x3d/facebook/x3d_s_13x6x1_facebook-kinetics400-rgb_20201027-623825a0.pth)\[1\] |
|         16x5x1          |  224x224   |  X3D_M   |     75.2     |     76.4     | 75.1 \[[SlowFast](https://github.com/facebookresearch/SlowFast/blob/master/MODEL_ZOO.md)\] | 76.2 \[[SlowFast](https://github.com/facebookresearch/SlowFast/blob/master/MODEL_ZOO.md)\] | [config](/configs/recognition/x3d/x3d_m_16x5x1_facebook-kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/x3d/facebook/x3d_m_16x5x1_facebook-kinetics400-rgb_20201027-3f42382a.pth)\[1\] |

\[1\] The models are ported from the repo [SlowFast](https://github.com/facebookresearch/SlowFast/) and tested on our data. Currently, we only support the testing of X3D models, training will be available soon.

1. The values in columns named after "reference" are the results got by testing the checkpoint released on the original repo and codes, using the same dataset with ours.
2. The validation set of Kinetics400 we used is same as the repo [SlowFast](https://github.com/facebookresearch/SlowFast/), which is available [here](https://github.com/facebookresearch/video-nonlocal-net/issues/67).

For more details on data preparation, you can refer to [Kinetics400](/tools/data/kinetics/README.md).

## Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test X3D model on Kinetics-400 dataset and dump the result to a pkl file.

```shell
python tools/test.py configs/recognition/x3d/x3d_s_13x6x1_facebook-kinetics400-rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --dump result.pkl
```

For more details, you can refer to the **Test** part in the [Training and Test Tutorial](/docs/en/user_guides/train_test.md).

## Citation

```BibTeX
@misc{feichtenhofer2020x3d,
      title={X3D: Expanding Architectures for Efficient Video Recognition},
      author={Christoph Feichtenhofer},
      year={2020},
      eprint={2004.04730},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
