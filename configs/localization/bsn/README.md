# BSN

[Bsn: Boundary sensitive network for temporal action proposal generation](https://openaccess.thecvf.com/content_ECCV_2018/html/Tianwei_Lin_BSN_Boundary_Sensitive_ECCV_2018_paper.html)

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

Temporal action proposal generation is an important yet challenging problem, since temporal proposals with rich action content are indispensable for analysing real-world videos with long duration and high proportion irrelevant content. This problem requires methods not only generating proposals with precise temporal boundaries, but also retrieving proposals to cover truth action instances with high recall and high overlap using relatively fewer proposals. To address these difficulties, we introduce an effective proposal generation method, named Boundary-Sensitive Network (BSN), which adopts "local to global" fashion. Locally, BSN first locates temporal boundaries with high probabilities, then directly combines these boundaries as proposals. Globally, with Boundary-Sensitive Proposal feature, BSN retrieves proposals by evaluating the confidence of whether a proposal contains an action within its region. We conduct experiments on two challenging datasets: ActivityNet-1.3 and THUMOS14, where BSN outperforms other state-of-the-art temporal action proposal generation methods with high recall and high temporal precision. Finally, further experiments demonstrate that by combining existing action classifiers, our method significantly improves the state-of-the-art temporal action detection performance.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/34324155/143016692-69efafbd-cec6-47f1-af45-371d0ff78a97.png" width="800"/>
</div>

## Results and Models

### ActivityNet feature

|    feature    | gpus | pretrain |  AUC  | AR@1  | AR@5  | AR@10 | AR@100 |   gpu_mem(M)    | iter time(s) |                   config                   |                   ckpt                   |                   log                    |
| :-----------: | :--: | :------: | :---: | :---: | :---: | :---: | :----: | :-------------: | :----------: | :----------------------------------------: | :--------------------------------------: | :--------------------------------------: |
| cuhk_mean_100 |  1   |   None   | 66.26 | 32.71 | 48.43 | 55.28 | 74.27  | 43(TEM)+25(PEM) |      -       | [config_TEM](/configs/localization/bsn/bsn_tem_1xb16-400x100-20e_activitynet-feature.py) [config_PGM](/configs/localization/bsn/bsn_pgm_400x100_activitynet-feature.py) [config_PEM](/configs/localization/bsn/bsn_pem_1xb16-400x100-20e_activitynet-feature.py) | [ckpt_TEM](https://download.openmmlab.com/mmaction/v1.0/localization/bsn/bsn_tem_1xb16-400x100-20e_activitynet-feature_20220908-9da79951.pth) [ckpt_PEM](https://download.openmmlab.com/mmaction/v1.0/localization/bsn/bsn_pem_1xb16-400x100-20e_activitynet-feature_20220908-ec2eb21d.pth) | [log_tem](https://download.openmmlab.com/mmaction/v1.0/localization/bsn/bsn_tem_1xb16-400x100-20e_activitynet-feature.log) [log_pem](https://download.openmmlab.com/mmaction/v1.0/localization/bsn/bsn_pem_1xb16-400x100-20e_activitynet-feature.log) |
| slowonly-k700 |  1   |   None   | 67.63 | 33.04 | 48.79 | 56.01 | 75.74  |        -        |      -       | [config_TEM](/configs/localization/bsn/bsn_tem_1xb16-2048x100-20e_activitynet-k700-feature.py) [config_PGM](/configs/localization/bsn/bsn_pgm_2048x100_activitynet-slowonly-k700-feature.py) [config_PEM](/configs/localization/bsn/bsn_pem_1xb16-2048x100-20e_activitynet-slowonly-k700-feature.py) | [ckpt_TEM](https://download.openmmlab.com/mmaction/v1.0/localization/bsn/bsn_tem_1xb16-2048x100-20e_activitynet-k700-feature_20230907-76069fda.pth) [ckpt_PEM](https://download.openmmlab.com/mmaction/v1.0/localization/bsn/bsn_pem_1xb16-2048x100-20e_activitynet-slowonly-k700-feature_20230907-44158b6d.pth) | [log_tem](https://download.openmmlab.com/mmaction/v1.0/localization/bsn/bsn_tem_1xb16-400x100-20e_activitynet-feature.log) [log_pem](https://download.openmmlab.com/mmaction/v1.0/localization/bsn/bsn_pem_1xb16-400x100-20e_activitynet-feature.log) |

1. The **gpus** indicates the number of gpu we used to get the checkpoint.
   According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you may set the learning rate proportional to the batch size if you use different GPUs or videos per GPU,
   e.g., lr=0.01 for 4 GPUs x 2 video/gpu and lr=0.08 for 16 GPUs x 4 video/gpu.
2. For feature column, cuhk_mean_100 denotes the widely used cuhk activitynet feature extracted by [anet2016-cuhk](https://github.com/yjxiong/anet2016-cuhk). The slowonly-k700 denotes the feature extracted using MMAction2's [SlowOnly model trained on Kinetics 700](/configs/recognition/slowonly/slowonly_imagenet-pretrained-r50_16xb16-8x8x1-steplr-150e_kinetics700-rgb.py). You can download this feature from [ActivityNet Data Preparation](/tools/data/activitynet/README.md).

For more details on data preparation, you can refer to [ActivityNet Data Preparation](/tools/data/activitynet/README.md).

## Training and Test

The traing of the BSN model is three-stages. We take the `cuhk_mean_100` feature as an example. For `slowonly-k700` feature, just need to replace the config file with the corresponding config file with `slowonly-k700` in the file name.

Firstly train the Temporal evaluation module (TEM):

```shell
python3 tools/train.py configs/localization/bsn/bsn_tem_1xb16-400x100-20e_activitynet-feature.py
```

After training use the TEM module to generate the probabilities sequence (actionness, starting, and ending) for the training and validation dataset:

```shell
python tools/test.py configs/localization/bsn/bsn_tem_1xb16-400x100-20e_activitynet-feature.py \
    work_dirs/bsn_400x100_20e_1xb16_activitynet_feature/tem_epoch_20.pth
```

The second step is to run the Proposal generation module (PGM) to generate Boundary-Sensitive Proposal (BSP) feature for the training and validation dataset:

```shell
python tools/misc/bsn_proposal_generation.py configs/localization/bsn/bsn_pgm_400x100_activitynet-feature.py --mode train
python tools/misc/bsn_proposal_generation.py configs/localization/bsn/bsn_pgm_400x100_activitynet-feature.py --mode test
```

The last step is to train (and validate) the Proposal evaluation module (PEM):

```shell
python python tools/train.py configs/localization/bsn/bsn_pem_1xb16-400x100-20e_activitynet-feature.py
```

(Optional) You can use the following command to generate a formatted proposal file, which will be fed into the action classifier (Currently supports only SSN and P-GCN, not including TSN, I3D etc.) to get the classification result of proposals.

```shell
python tools/data/activitynet/convert_proposal_format.py
```

## Citation

```BibTeX
@inproceedings{lin2018bsn,
  title={Bsn: Boundary sensitive network for temporal action proposal generation},
  author={Lin, Tianwei and Zhao, Xu and Su, Haisheng and Wang, Chongjing and Yang, Ming},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={3--19},
  year={2018}
}
```
