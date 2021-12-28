# I3D

## Abstract

<!-- [ABSTRACT] -->

The paucity of videos in current action classification datasets (UCF-101 and HMDB-51) has made it difficult to identify good video architectures, as most methods obtain similar performance on existing small-scale benchmarks. This paper re-evaluates state-of-the-art architectures in light of the new Kinetics Human Action Video dataset. Kinetics has two orders of magnitude more data, with 400 human action classes and over 400 clips per class, and is collected from realistic, challenging YouTube videos. We provide an analysis on how current architectures fare on the task of action classification on this dataset and how much performance improves on the smaller benchmark datasets after pre-training on Kinetics. We also introduce a new Two-Stream Inflated 3D ConvNet (I3D) that is based on 2D ConvNet inflation: filters and pooling kernels of very deep image classification ConvNets are expanded into 3D, making it possible to learn seamless spatio-temporal feature extractors from video while leveraging successful ImageNet architecture designs and even their parameters. We show that, after pre-training on Kinetics, I3D models considerably improve upon the state-of-the-art in action classification, reaching 80.9% on HMDB-51 and 98.0% on UCF-101.

<!-- [IMAGE] -->
<div align=center>
<img src="https://user-images.githubusercontent.com/34324155/143043624-1944704a-5d3e-4a3f-b258-1505c49f6092.png" width="800"/>
</div>

## Citation

<!-- [ALGORITHM] -->

```BibTeX
@inproceedings{inproceedings,
  author = {Carreira, J. and Zisserman, Andrew},
  year = {2017},
  month = {07},
  pages = {4724-4733},
  title = {Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset},
  doi = {10.1109/CVPR.2017.502}
}
```

<!-- [BACKBONE] -->

```BibTeX
@article{NonLocal2018,
  author =   {Xiaolong Wang and Ross Girshick and Abhinav Gupta and Kaiming He},
  title =    {Non-local Neural Networks},
  journal =  {CVPR},
  year =     {2018}
}
```

## Model Zoo

### Kinetics-400

|config | resolution | gpus | backbone |pretrain| top1 acc| top5 acc | inference_time(video/s) | gpu_mem(M)| ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|[i3d_r50_32x2x1_100e_kinetics400_rgb](/configs/recognition/i3d/i3d_r50_32x2x1_100e_kinetics400_rgb.py) |340x256|8| ResNet50|ImageNet |72.68|90.78|1.7 (320x3 frames)| 5170|[ckpt](https://download.openmmlab.com/mmaction/recognition/i3d/i3d_r50_32x2x1_100e_kinetics400_rgb/i3d_r50_32x2x1_100e_kinetics400_rgb_20200614-c25ef9a4.pth) | [log](https://download.openmmlab.com/mmaction/recognition/i3d/i3d_r50_32x2x1_100e_kinetics400_rgb/20200614_060456.log)| [json](https://download.openmmlab.com/mmaction/recognition/i3d/i3d_r50_32x2x1_100e_kinetics400_rgb/20200614_060456.log.json)|
|[i3d_r50_32x2x1_100e_kinetics400_rgb](/configs/recognition/i3d/i3d_r50_32x2x1_100e_kinetics400_rgb.py) |short-side 256|8| ResNet50|ImageNet | 73.27|90.92|x|5170|[ckpt](https://download.openmmlab.com/mmaction/recognition/i3d/i3d_r50_256p_32x2x1_100e_kinetics400_rgb/i3d_r50_256p_32x2x1_100e_kinetics400_rgb_20200801-7d9f44de.pth)|[log](https://download.openmmlab.com/mmaction/recognition/i3d/i3d_r50_256p_32x2x1_100e_kinetics400_rgb/20200725_031555.log)|[json](https://download.openmmlab.com/mmaction/recognition/i3d/i3d_r50_256p_32x2x1_100e_kinetics400_rgb/20200725_031555.log.json)|
|[i3d_r50_video_32x2x1_100e_kinetics400_rgb](/configs/recognition/i3d/i3d_r50_video_32x2x1_100e_kinetics400_rgb.py)|short-side 256p|8| ResNet50 |ImageNet|72.85 |90.75 |x|5170|[ckpt](https://download.openmmlab.com/mmaction/recognition/i3d/i3d_r50_video_32x2x1_100e_kinetics400_rgb/i3d_r50_video_32x2x1_100e_kinetics400_rgb_20200826-e31c6f52.pth)|[log](https://download.openmmlab.com/mmaction/recognition/i3d/i3d_r50_video_32x2x1_100e_kinetics400_rgb/20200706_143014.log)|[json](https://download.openmmlab.com/mmaction/recognition/i3d/i3d_r50_video_32x2x1_100e_kinetics400_rgb/20200706_143014.log.json)|
|[i3d_r50_dense_32x2x1_100e_kinetics400_rgb](/configs/recognition/i3d/i3d_r50_dense_32x2x1_100e_kinetics400_rgb.py) |340x256|8x2| ResNet50| ImageNet|72.77|90.57|1.7 (320x3 frames)| 5170| [ckpt](https://download.openmmlab.com/mmaction/recognition/i3d/i3d_r50_dense_32x2x1_100e_kinetics400_rgb/i3d_r50_dense_32x2x1_100e_kinetics400_rgb_20200616-2bbb4361.pth) | [log](https://download.openmmlab.com/mmaction/recognition/i3d/i3d_r50_dense_32x2x1_100e_kinetics400_rgb/20200616_230011.log)| [json](https://download.openmmlab.com/mmaction/recognition/i3d/i3d_r50_dense_32x2x1_100e_kinetics400_rgb/20200616_230011.log.json)|
|[i3d_r50_dense_32x2x1_100e_kinetics400_rgb](/configs/recognition/i3d/i3d_r50_dense_32x2x1_100e_kinetics400_rgb.py) |short-side 256|8| ResNet50| ImageNet|73.48|91.00|x| 5170| [ckpt](https://download.openmmlab.com/mmaction/recognition/i3d/i3d_r50_dense_256p_32x2x1_100e_kinetics400_rgb/i3d_r50_dense_256p_32x2x1_100e_kinetics400_rgb_20200725-24eb54cc.pth)|[log](https://download.openmmlab.com/mmaction/recognition/i3d/i3d_r50_dense_256p_32x2x1_100e_kinetics400_rgb/20200725_031604.log)|[json](https://download.openmmlab.com/mmaction/recognition/i3d/i3d_r50_dense_256p_32x2x1_100e_kinetics400_rgb/20200725_031604.log.json)|
|[i3d_r50_lazy_32x2x1_100e_kinetics400_rgb](/configs/recognition/i3d/i3d_r50_lazy_32x2x1_100e_kinetics400_rgb.py) |340x256|8| ResNet50 |ImageNet|72.32|90.72|1.8 (320x3 frames)| 5170| [ckpt](https://download.openmmlab.com/mmaction/recognition/i3d/i3d_r50_fast_32x2x1_100e_kinetics400_rgb/i3d_r50_fast_32x2x1_100e_kinetics400_rgb_20200612-000e4d2a.pth) | [log](https://download.openmmlab.com/mmaction/recognition/i3d/i3d_r50_fast_32x2x1_100e_kinetics400_rgb/20200612_233836.log)| [json](https://download.openmmlab.com/mmaction/recognition/i3d/i3d_r50_fast_32x2x1_100e_kinetics400_rgb/20200612_233836.log.json)|
|[i3d_r50_lazy_32x2x1_100e_kinetics400_rgb](/configs/recognition/i3d/i3d_r50_lazy_32x2x1_100e_kinetics400_rgb.py) |short-side 256|8| ResNet50| ImageNet|73.24|90.99|x| 5170| [ckpt](https://download.openmmlab.com/mmaction/recognition/i3d/i3d_r50_fast_256p_32x2x1_100e_kinetics400_rgb/i3d_r50_fast_256p_32x2x1_100e_kinetics400_rgb_20200817-4e90d1d5.pth)| [log](https://download.openmmlab.com/mmaction/recognition/i3d/i3d_r50_fast_256p_32x2x1_100e_kinetics400_rgb/20200725_031457.log) | [json](https://download.openmmlab.com/mmaction/recognition/i3d/i3d_r50_fast_256p_32x2x1_100e_kinetics400_rgb/20200725_031457.log.json) |
|[i3d_nl_embedded_gaussian_r50_32x2x1_100e_kinetics400_rgb](/configs/recognition/i3d/i3d_nl_embedded_gaussian_r50_32x2x1_100e_kinetics400_rgb.py)|short-side 256p|8x4| ResNet50 |ImageNet|74.71|91.81|x|6438|[ckpt](https://download.openmmlab.com/mmaction/recognition/i3d/i3d_nl_embedded_gaussian_r50_32x2x1_100e_kinetics400_rgb/i3d_nl_embedded_gaussian_r50_32x2x1_100e_kinetics400_rgb_20200813-6e6aef1b.pth)|[log](https://download.openmmlab.com/mmaction/recognition/i3d/i3d_nl_embedded_gaussian_r50_32x2x1_100e_kinetics400_rgb/20200813_034054.log)|[json](https://download.openmmlab.com/mmaction/recognition/i3d/i3d_nl_embedded_gaussian_r50_32x2x1_100e_kinetics400_rgb/20200813_034054.log.json)|
|[i3d_nl_gaussian_r50_32x2x1_100e_kinetics400_rgb](/configs/recognition/i3d/i3d_nl_gaussian_r50_32x2x1_100e_kinetics400_rgb.py)|short-side 256p|8x4| ResNet50 |ImageNet|73.37|91.26|x|4944|[ckpt](https://download.openmmlab.com/mmaction/recognition/i3d/i3d_nl_gaussian_r50_32x2x1_100e_kinetics400_rgb/i3d_nl_gaussian_r50_32x2x1_100e_kinetics400_rgb_20200815-17f84aa2.pth)|[log](https://download.openmmlab.com/mmaction/recognition/i3d/i3d_nl_gaussian_r50_32x2x1_100e_kinetics400_rgb/20200813_034909.log)|[json](https://download.openmmlab.com/mmaction/recognition/i3d/i3d_nl_gaussian_r50_32x2x1_100e_kinetics400_rgb/20200813_034909.log.json)|
|[i3d_nl_dot_product_r50_32x2x1_100e_kinetics400_rgb](/configs/recognition/i3d/i3d_nl_dot_product_r50_32x2x1_100e_kinetics400_rgb.py)|short-side 256p|8x4| ResNet50 |ImageNet|73.92|91.59|x|4832|[ckpt](https://download.openmmlab.com/mmaction/recognition/i3d/i3d_nl_dot_product_r50_32x2x1_100e_kinetics400_rgb/i3d_nl_dot_product_r50_32x2x1_100e_kinetics400_rgb_20200814-7c30d5bb.pth)|[log](https://download.openmmlab.com/mmaction/recognition/i3d/i3d_nl_dot_product_r50_32x2x1_100e_kinetics400_rgb/20200814_044208.log)|[json](https://download.openmmlab.com/mmaction/recognition/i3d/i3d_nl_dot_product_r50_32x2x1_100e_kinetics400_rgb/20200814_044208.log.json)|

:::{note}

1. The **gpus** indicates the number of gpu we used to get the checkpoint. It is noteworthy that the configs we provide are used for 8 gpus as default.
   According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you may set the learning rate proportional to the batch size if you use different GPUs or videos per GPU,
   e.g., lr=0.01 for 4 GPUs x 2 video/gpu and lr=0.08 for 16 GPUs x 4 video/gpu.
2. The **inference_time** is got by this [benchmark script](/tools/analysis/benchmark.py), where we use the sampling frames strategy of the test setting and only care about the model inference time, not including the IO time and pre-processing time. For each setting, we use 1 gpu and set batch size (videos per gpu) to 1 to calculate the inference time.
3. The validation set of Kinetics400 we used consists of 19796 videos. These videos are available at [Kinetics400-Validation](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136485_link_cuhk_edu_hk/EbXw2WX94J1Hunyt3MWNDJUBz-nHvQYhO9pvKqm6g39PMA?e=a9QldB). The corresponding [data list](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_val_list.txt) (each line is of the format 'video_id, num_frames, label_index') and the [label map](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_class2ind.txt) are also available.

:::

For more details on data preparation, you can refer to Kinetics400 in [Data Preparation](/docs/data_preparation.md).

## Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train I3D model on Kinetics-400 dataset in a deterministic option with periodic validation.

```shell
python tools/train.py configs/recognition/i3d/i3d_r50_32x2x1_100e_kinetics400_rgb.py \
    --work-dir work_dirs/i3d_r50_32x2x1_100e_kinetics400_rgb \
    --validate --seed 0 --deterministic
```

For more details, you can refer to **Training setting** part in [getting_started](/docs/getting_started.md#training-setting).

## Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test I3D model on Kinetics-400 dataset and dump the result to a json file.

```shell
python tools/test.py configs/recognition/i3d/i3d_r50_32x2x1_100e_kinetics400_rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --eval top_k_accuracy mean_class_accuracy \
    --out result.json --average-clips prob
```

For more details, you can refer to **Test a dataset** part in [getting_started](/docs/getting_started.md#test-a-dataset).
