# Omnisource

<!-- TODO: add links to the tech report -->

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

We propose to train a recognizer that can classify images and videos. The recognizer is jointly trained on image and video datasets. Compared with pre-training on the same image dataset, this method can significantly improve the video recognition performance.

<!-- [IMAGE]

<div align=center>
<img src="https://user-images.githubusercontent.com/34324155/143044111-94676f64-7ba8-4081-9011-f8054bed7030.png" width="800"/>
</div>
-->

## Results and Models

### Kinetics-400

| frame sampling strategy |   scheduler   | resolution | gpus | backbone | joint-training | top1 acc | top5 acc | testing protocol  | FLOPs  | params |            config             |            ckpt             |             log             |
| :---------------------: | :-----------: | :--------: | :--: | :------: | :------------: | :------: | :------: | :---------------: | :----: | :----: | :---------------------------: | :-------------------------: | :-------------------------: |
|          8x8x1          | Linear+Cosine |  224x224   |  8   | ResNet50 |    ImageNet    |  77.30   |  93.23   | 10 clips x 3 crop | 54.75G | 32.45M | [config](/configs/recognition/omnisource/slowonly_r50_8xb16-8x8x1-256e_imagenet-kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/omnisource/slowonly_r50_8xb16-8x8x1-256e_imagenet-kinetics400-rgb_20230208-61c4be0d.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/omnisource/slowonly_r50_8xb16-8x8x1-256e_imagenet-kinetics400-rgb.log) |

1. The **gpus** indicates the number of gpus we used to get the checkpoint. If you want to use a different number of gpus or videos per gpu, the best way is to set `--auto-scale-lr` when calling `tools/train.py`, this parameter will auto-scale the learning rate according to the actual batch size and the original batch size.
2. The validation set of Kinetics400 we used consists of 19796 videos. These videos are available at [Kinetics400-Validation](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136485_link_cuhk_edu_hk/EbXw2WX94J1Hunyt3MWNDJUBz-nHvQYhO9pvKqm6g39PMA?e=a9QldB). The corresponding [data list](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_val_list.txt) (each line is of the format 'video_id, num_frames, label_index') and the [label map](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_class2ind.txt) are also available.

For more details on data preparation, you can refer to [Kinetics400](/tools/data/kinetics/README.md).

## Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train SlowOnly model on Kinetics-400 dataset in a deterministic option with periodic validation.

```shell
python tools/train.py configs/recognition/omnisource/slowonly_r50_8xb16-8x8x1-256e_imagenet-kinetics400-rgb.py \
    --seed=0 --deterministic
```

We found that the training of this Omnisource model could crash for unknown reasons. If this happens, you can resume training by adding the `--cfg-options resume=True` to the training script.

For more details, you can refer to the **Training** part in the [Training and Test Tutorial](/docs/en/user_guides/train_test.md).

## Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test SlowOnly model on Kinetics-400 dataset and dump the result to a pkl file.

```shell
python tools/test.py configs/recognition/omnisource/slowonly_r50_8xb16-8x8x1-256e_imagenet-kinetics400-rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --dump result.pkl
```

For more details, you can refer to the **Test** part in the [Training and Test Tutorial](/docs/en/user_guides/train_test.md).

## Citation

```BibTeX
@inproceedings{feichtenhofer2019slowfast,
  title={Slowfast networks for video recognition},
  author={Feichtenhofer, Christoph and Fan, Haoqi and Malik, Jitendra and He, Kaiming},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  pages={6202--6211},
  year={2019}
}
```

```BibTeX
@article{duan2020omni,
  title={Omni-sourced Webly-supervised Learning for Video Recognition},
  author={Duan, Haodong and Zhao, Yue and Xiong, Yuanjun and Liu, Wentao and Lin, Dahua},
  journal={arXiv preprint arXiv:2003.13042},
  year={2020}
}
```
