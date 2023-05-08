# VideoMAE V2

[VideoMAE V2: Scaling Video Masked Autoencoders with Dual Masking](https://arxiv.org/abs/2303.16727)

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

Scale is the primary factor for building a powerful foundation model that could well generalize to a variety of downstream tasks. However, it is still challenging to train video foundation models with billions of parameters. This paper shows that video masked autoencoder (VideoMAE) is a scalable and general self-supervised pre-trainer for building video foundation models. We scale the VideoMAE in both model and data with a core design. Specifically, we present a dual masking strategy for efficient pre-training, with an encoder operating on a subset of video tokens and a decoder processing another subset of video tokens. Although VideoMAE is very efficient due to high masking ratio in encoder, masking decoder can still further reduce the overall computational cost. This enables the efficient pre-training of billion-level models in video. We also use a progressive training paradigm that involves an initial pre-training on a diverse multi-sourced unlabeled dataset, followed by a post-pre-training on a mixed labeled dataset. Finally, we successfully train a video ViT model with a billion parameters, which achieves a new state-of-the-art performance on the datasets of Kinetics (90.0% on K400 and 89.9% on K600) and Something-Something (68.7% on V1 and 77.0% on V2). In addition, we extensively verify the pre-trained video ViT models on a variety of downstream tasks, demonstrating its effectiveness as a general video representation learner.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/35596075/237352561-6204d743-8705-43f5-817f-0bc4907b88d0.png" width="800"/>
</div>

## Results and Models

### Kinetics-400

| frame sampling strategy |   resolution   | backbone | top1 acc | top5 acc |         reference top1 acc         |         reference top5 acc         | testing protocol  | FLOPs | params |         config         |         ckpt          |
| :---------------------: | :------------: | :------: | :------: | :------: | :--------------------------------: | :--------------------------------: | :---------------: | :---: | :----: | :--------------------: | :-------------------: |
|         16x4x1          | short-side 320 |  ViT-S   |   83.6   |   96.3   | 83.7 \[[VideoMAE V2](https://github.com/OpenGVLab/VideoMAEv2/blob/master/docs/MODEL_ZOO.md)\] | 96.2 \[[VideoMAE V2](https://github.com/OpenGVLab/VideoMAEv2/blob/master/docs/MODEL_ZOO.md)\] | 5 clips x 3 crops |  57G  |  22M   | [config](/configs/recognition/videomaev2/vit-small-p16_videomaev2-vit-g-dist-k710-pre_16x4x1_kinetics-400.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/videomaev2/vit-small-p16_videomaev2-vit-g-dist-k710-pre_16x4x1_kinetics-400/vit-small-p16_videomaev2-vit-g-dist-k710-pre_16x4x1_kinetics-400_20230510-25c748fd.pth) \[1\] |
|         16x4x1          | short-side 320 |  ViT-B   |   86.6   |   97.3   | 86.6 \[[VideoMAE V2](https://github.com/OpenGVLab/VideoMAEv2/blob/master/docs/MODEL_ZOO.md)\] | 97.3 \[[VideoMAE V2](https://github.com/OpenGVLab/VideoMAEv2/blob/master/docs/MODEL_ZOO.md)\] | 5 clips x 3 crops | 180G  |  87M   | [config](/configs/recognition/videomaev2/vit-base-p16_videomaev2-vit-g-dist-k710-pre_16x4x1_kinetics-400.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/videomaev2/vit-base-p16_videomaev2-vit-g-dist-k710-pre_16x4x1_kinetics-400/vit-base-p16_videomaev2-vit-g-dist-k710-pre_16x4x1_kinetics-400_20230510-3e7f93b2.pth) \[1\] |

\[1\] The models were distilled from the VideoMAE V2-g model. Specifically, models are initialized with VideoMAE V2 pretraining, then distilled on Kinetics 710 dataset. They are ported from the repo [VideoMAE V2](https://github.com/OpenGVLab/VideoMAEv2) and tested on our data. The VideoMAE V2-g model can be obtained from the original repository. Currently, we only support the testing of VideoMAE V2 models.

1. The values in columns named after "reference" are the results of the original repo.
2. The validation set of Kinetics400 we used consists of 19796 videos. These videos are available at [Kinetics400-Validation](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136485_link_cuhk_edu_hk/EbXw2WX94J1Hunyt3MWNDJUBz-nHvQYhO9pvKqm6g39PMA?e=a9QldB). The corresponding [data list](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_val_list.txt) (each line is of the format 'video_id, num_frames, label_index') and the [label map](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_class2ind.txt) are also available.

For more details on data preparation, you can refer to [preparing_kinetics](/tools/data/kinetics/README.md).

## Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test ViT-base model on Kinetics-400 dataset and dump the result to a pkl file.

```shell
python tools/test.py configs/recognition/videomaev2/vit-base-p16_videomaev2-vit-g-dist-k710-pre_16x4x1_kinetics-400.py \
    checkpoints/SOME_CHECKPOINT.pth --dump result.pkl
```

For more details, you can refer to the **Test** part in the [Training and Test Tutorial](/docs/en/user_guides/train_test.md).

## Citation

```BibTeX
@misc{wang2023videomaev2,
      title={VideoMAE V2: Scaling Video Masked Autoencoders with Dual Masking},
      author={Limin Wang and Bingkun Huang and Zhiyu Zhao and Zhan Tong and Yinan He and Yi Wang and Yali Wang and Yu Qiao},
      year={2023},
      eprint={2303.16727},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
