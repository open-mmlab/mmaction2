# VideoMAE

[VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training](https://arxiv.org/abs/2203.12602)

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

Pre-training video transformers on extra large-scale datasets is generally required to achieve premier performance on relatively small datasets. In this paper, we show that video masked autoencoders (VideoMAE) are data-efficient learners for self-supervised video pre-training (SSVP). We are inspired by the recent ImageMAE and propose customized video tube masking with an extremely high ratio. This simple design makes video reconstruction a more challenging self-supervision task, thus encouraging extracting more effective video representations during this pre-training process. We obtain three important findings on SSVP: (1) An extremely high proportion of masking ratio (i.e., 90% to 95%) still yields favorable performance of VideoMAE. The temporally redundant video content enables a higher masking ratio than that of images. (2) VideoMAE achieves impressive results on very small datasets (i.e., around 3k-4k videos) without using any extra data. (3) VideoMAE shows that data quality is more important than data quantity for SSVP. Domain shift between pre-training and target datasets is an important issue. Notably, our VideoMAE with the vanilla ViT can achieve 87.4% on Kinetics-400, 75.4% on Something-Something V2, 91.3% on UCF101, and 62.6% on HMDB51, without using any extra data.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/35267818/191656296-14f28f4a-203f-4eeb-a4c3-c2efdb6d1ab4.png" width="800"/>
</div>

## Results and Models

### Kinetics-400

| frame sampling strategy |   resolution   | backbone | top1 acc | top5 acc |         reference top1 acc         |         reference top5 acc         | testing protocol  | FLOPs | params |         config         |         ckpt          |
| :---------------------: | :------------: | :------: | :------: | :------: | :--------------------------------: | :--------------------------------: | :---------------: | :---: | :----: | :--------------------: | :-------------------: |
|         16x4x1          | short-side 320 |  ViT-B   |   81.3   |   95.0   | 81.5 \[[VideoMAE](https://github.com/MCG-NJU/VideoMAE/blob/main/MODEL_ZOO.md)\] | 95.1 \[[VideoMAE](https://github.com/MCG-NJU/VideoMAE/blob/main/MODEL_ZOO.md)\] | 5 clips x 3 crops | 180G  |  87M   | [config](/configs/recognition/videomae/vit-base-p16_videomae-k400-pre_16x4x1_kinetics-400.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/videomae/vit-base-p16_videomae-k400-pre_16x4x1_kinetics-400_20221013-860a3cd3.pth) \[1\] |
|         16x4x1          | short-side 320 |  ViT-L   |   85.3   |   96.7   | 85.2 \[[VideoMAE](https://github.com/MCG-NJU/VideoMAE/blob/main/MODEL_ZOO.md)\] | 96.8 \[[VideoMAE](https://github.com/MCG-NJU/VideoMAE/blob/main/MODEL_ZOO.md)\] | 5 clips x 3 crops | 597G  |  305M  | [config](/configs/recognition/videomae/vit-large-p16_videomae-k400-pre_16x4x1_kinetics-400.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/videomae/vit-large-p16_videomae-k400-pre_16x4x1_kinetics-400_20221013-229dbb03.pth) \[1\] |

\[1\] The models are ported from the repo [VideoMAE](https://github.com/MCG-NJU/VideoMAE) and tested on our data. Currently, we only support the testing of VideoMAE models, training will be available soon.

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
python tools/test.py configs/recognition/videomae/vit-base-p16_videomae-k400-pre_16x4x1_kinetics-400.py \
    checkpoints/SOME_CHECKPOINT.pth --dump result.pkl
```

For more details, you can refer to the **Test** part in the [Training and Test Tutorial](/docs/en/user_guides/train_test.md).

## Citation

```BibTeX
@inproceedings{tong2022videomae,
  title={Video{MAE}: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training},
  author={Zhan Tong and Yibing Song and Jue Wang and Limin Wang},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}
```
