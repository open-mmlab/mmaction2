# LFB

## Introduction

[ALGORITHM]

```BibTeX
@inproceedings{wu2019long,
  title={Long-term feature banks for detailed video understanding},
  author={Wu, Chao-Yuan and Feichtenhofer, Christoph and Fan, Haoqi and He, Kaiming and Krahenbuhl, Philipp and Girshick, Ross},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={284--293},
  year={2019}
}
```

## Model Zoo

### AVA2.1

| Model | Modality |  Pretrained  | Backbone  | Input | gpus |   Resolution   | mAP  | log | json | ckpt |
| :----------------------------------------------------------: | :------: | :----------: | :-------: | :---: | :--: | :------------: | :--: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [lfb_nl_kinetics_pretrained_slowonly_r50_4x16x1_20e_ava_rgb.py](/configs/detection/lfb/lfb_nl_kinetics_pretrained_slowonly_r50_4x16x1_20e_ava_rgb.py) |   RGB    | Kinetics-400 | SlowOnly_R50 | 4x16 | 8 | short-side 256 | 24.11 | [log](waiting for url) | [json](waiting for url) | [ckpt](waiting for url) |

- Notes:

1. The **gpus** indicates the number of gpu we used to get the checkpoint.
   According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you may set the learning rate proportional to the batch size if you use different GPUs or videos per GPU,
   e.g., lr=0.01 for 4 GPUs x 2 video/gpu and lr=0.08 for 16 GPUs x 4 video/gpu.
