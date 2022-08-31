# C2D

<!-- [ALGORITHM] -->
C2D is the baseline of [Non-local Neural Networks](https://arxiv.org/abs/1711.07971)

NOTICE: C2D implementations are slightly different between 1.The paper above; 2."SlowFast" repo; 3."Video-Nonlocal-Net" repo.

C2D implementation in MMAction2 is kept same as the ["Video-Nonlocal-Net" repo](https://github.com/facebookresearch/video-nonlocal-net/tree/main/scripts/run_c2d_baseline_400k.sh)

Specifically:
- maxpool3d_1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 0, 0))
- maxpool3d_2 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))


C2D_Nopool implementation in MMAction2 is kept same as the ["SlowFast" repo](https://github.com/facebookresearch/SlowFast/blob/main/configs/Kinetics/c2/C2D_NOPOOL_8x8_R50.yaml)


<!-- [ABSTRACT] -->


<!-- [IMAGE] -->



## Results and Models

### Kinetics-400

| config                                                                                                             |   resolution   | gpus  | backbone | pretrain | top1 acc | top5 acc |                                        reference top1 acc                                        |                                        reference top5 acc                                        | inference_time(video/s) | gpu_mem(M) |   ckpt   |   log   |   json   |
| :----------------------------------------------------------------------------------------------------------------- | :------------: | :---: | :------: | :------: | :------: | :------: | :----------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------: | :---------------------: | :--------: | :------: | :-----: | :------: |
| [c2d_nopool_r50_8x8x1_100e_kinetics400_rgb](/configs/recognition/c2d/c2d_nopool_r50_8x8x1_100e_kinetics400_rgb.py) | short-side 256 |   8   | ResNet50 | ImageNet |  70.53   |  89.26   | [67.2](https://github.com/facebookresearch/SlowFast/blob/main/MODEL_ZOO.md#kinetics-400-and-600) | [87.8](https://github.com/facebookresearch/SlowFast/blob/main/MODEL_ZOO.md#kinetics-400-and-600) |            x            |   21548    | [ckpt]() | [log]() | [json]() |
| [c2d_r50_8x8x1_100e_kinetics400_rgb](/configs/recognition/c2d/c2d_r50_8x8x1_100e_kinetics400_rgb.py)               | short-side 256 |   8   | ResNet50 | ImageNet |  71.95   |  89.82   | [71.9](https://github.com/facebookresearch/video-nonlocal-net#modifications-for-improving-speed) | [90.0](https://github.com/facebookresearch/video-nonlocal-net#modifications-for-improving-speed) |            x            |   16961    | [ckpt]() | [log]() | [json]() |

:::{note}

1. The **gpus** indicates the number of gpu we used to get the checkpoint. It is noteworthy that the configs we provide are used for 8 gpus as default.
   According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you may set the learning rate proportional to the batch size if you use different GPUs or videos per GPU,
   e.g., lr=0.01 for 4 GPUs x 2 video/gpu and lr=0.08 for 16 GPUs x 4 video/gpu.
2. The **inference_time** is got by this [benchmark script](/tools/analysis/benchmark.py), where we use the sampling frames strategy of the test setting and only care about the model inference time,
   not including the IO time and pre-processing time. For each setting, we use 1 gpu and set batch size (videos per gpu) to 1 to calculate the inference time.
3. The values in columns named after "reference" are the results got by training on the original repo, using the same model settings.
4. The validation set of Kinetics400 we used consists of 19796 videos. These videos are available at [Kinetics400-Validation](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136485_link_cuhk_edu_hk/EbXw2WX94J1Hunyt3MWNDJUBz-nHvQYhO9pvKqm6g39PMA?e=a9QldB). The corresponding [data list](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_val_list.txt) (each line is of the format 'video_id, num_frames, label_index') and the [label map](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_class2ind.txt) are also available.

:::

For more details on data preparation, you can refer to

- [preparing_ucf101](/tools/data/ucf101/README.md)
- [preparing_kinetics](/tools/data/kinetics/README.md)
- [preparing_sthv1](/tools/data/sthv1/README.md)
- [preparing_sthv2](/tools/data/sthv2/README.md)
- [preparing_mit](/tools/data/mit/README.md)
- [preparing_mmit](/tools/data/mmit/README.md)
- [preparing_hvu](/tools/data/hvu/README.md)
- [preparing_hmdb51](/tools/data/hmdb51/README.md)

## Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

<!-- Example: train TSN model on Kinetics-400 dataset in a deterministic option with periodic validation.

```shell
python tools/train.py configs/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb.py \
    --work-dir work_dirs/tsn_r50_1x1x3_100e_kinetics400_rgb \
    --validate --seed 0 --deterministic
``` -->

For more details, you can refer to **Training setting** part in [getting_started](/docs/getting_started.md#training-setting).

## Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

<!-- Example: test TSN model on Kinetics-400 dataset and dump the result to a json file.

```shell
python tools/test.py configs/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --eval top_k_accuracy mean_class_accuracy \
    --out result.json
``` -->

For more details, you can refer to **Test a dataset** part in [getting_started](/docs/getting_started.md#test-a-dataset).

## Citation

```BibTeX
@article{XiaolongWang2017NonlocalNN,
  title={Non-local Neural Networks},
  author={Xiaolong Wang and Ross Girshick and Abhinav Gupta and Kaiming He},
  journal={arXiv: Computer Vision and Pattern Recognition},
  year={2017}
}
```
