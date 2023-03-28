# Action Localization Models

## BMN

[Bmn: Boundary-matching network for temporal action proposal generation](https://openaccess.thecvf.com/content_ICCV_2019/html/Lin_BMN_Boundary-Matching_Network_for_Temporal_Action_Proposal_Generation_ICCV_2019_paper.html)

<!-- [ALGORITHM] -->

### Abstract

<!-- [ABSTRACT] -->

Temporal action proposal generation is an challenging and promising task which aims to locate temporal regions in real-world videos where action or event may occur. Current bottom-up proposal generation methods can generate proposals with precise boundary, but cannot efficiently generate adequately reliable confidence scores for retrieving proposals. To address these difficulties, we introduce the Boundary-Matching (BM) mechanism to evaluate confidence scores of densely distributed proposals, which denote a proposal as a matching pair of starting and ending boundaries and combine all densely distributed BM pairs into the BM confidence map. Based on BM mechanism, we propose an effective, efficient and end-to-end proposal generation method, named Boundary-Matching Network (BMN), which generates proposals with precise temporal boundaries as well as reliable confidence scores simultaneously. The two-branches of BMN are jointly trained in an unified framework. We conduct experiments on two challenging datasets: THUMOS-14 and ActivityNet-1.3, where BMN shows significant performance improvement with remarkable efficiency and generalizability. Further, combining with existing action classifier, BMN can achieve state-of-the-art temporal action detection performance.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/34324155/143016479-2ca7e8b6-a17b-4a4c-b4c9-ae731935cd91.png" width="800"/>
</div>

### Results and Models

#### ActivityNet feature

|    feature    | gpus | pretrain |  AUC  | AR@1  | AR@5  | AR@10 | AR@100 | gpu_mem(M) | iter time(s) |                    config                    |                    ckpt                    |                    log                    |
| :-----------: | :--: | :------: | :---: | :---: | :---: | :---: | :----: | :--------: | :----------: | :------------------------------------------: | :----------------------------------------: | :---------------------------------------: |
| cuhk_mean_100 |  2   |   None   | 67.25 | 32.89 | 49.43 | 56.64 | 75.29  |    5412    |      -       | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/localization/bmn/bmn_2xb8-400x100-9e_activitynet-feature.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/localization/bmn/bmn_2xb8-400x100-9e_activitynet-feature_20220908-79f92857.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/localization/bmn/bmn_2xb8-400x100-9e_activitynet-feature.log) |

1. The **gpus** indicates the number of gpu we used to get the checkpoint.
   According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you may set the learning rate proportional to the batch size if you use different GPUs or videos per GPU,
   e.g., lr=0.01 for 4 GPUs x 2 video/gpu and lr=0.08 for 16 GPUs x 4 video/gpu.
2. For feature column, cuhk_mean_100 denotes the widely used cuhk activitynet feature extracted by [anet2016-cuhk](https://github.com/yjxiong/anet2016-cuhk).
3. We evaluate the action detection performance of BMN, using  [anet_cuhk_2017](https://download.openmmlab.com/mmaction/localization/cuhk_anet17_pred.json) submission for ActivityNet2017 Untrimmed Video Classification Track to assign label for each action proposal.

\*We train BMN with the [official repo](https://github.com/JJBOY/BMN-Boundary-Matching-Network), evaluate its proposal generation and action detection performance with [anet_cuhk_2017](https://download.openmmlab.com/mmaction/localization/cuhk_anet17_pred.json) for label assigning.

For more details on data preparation, you can refer to [ActivityNet Data Preparation](https://github.com/open-mmlab/mmaction2/tree/master/tools/data/activitynet/README.md).

### Train

Train BMN model on ActivityNet features dataset.

```shell
bash tools/dist_train.sh configs/localization/bmn/bmn_2xb8-400x100-9e_activitynet-feature.py 2
```

For more details, you can refer to the **Training** part in the [Training and Test Tutorial](en/user_guides/4_train_test.md).

### Test

Test BMN on ActivityNet feature dataset.

```shell
python3 tools/test.py  configs/localization/bmn/bmn_2xb8-400x100-9e_activitynet-feature.py CHECKPOINT.PTH
```

For more details, you can refer to the **Testing** part in the [Training and Test Tutorial](en/user_guides/4_train_test.md).

### Citation

```BibTeX
@inproceedings{lin2019bmn,
  title={Bmn: Boundary-matching network for temporal action proposal generation},
  author={Lin, Tianwei and Liu, Xiao and Li, Xin and Ding, Errui and Wen, Shilei},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={3889--3898},
  year={2019}
}
```

<!-- [DATASET] -->

```BibTeX
@article{zhao2017cuhk,
  title={Cuhk \& ethz \& siat submission to activitynet challenge 2017},
  author={Zhao, Y and Zhang, B and Wu, Z and Yang, S and Zhou, L and Yan, S and Wang, L and Xiong, Y and Lin, D and Qiao, Y and others},
  journal={arXiv preprint arXiv:1710.08011},
  volume={8},
  year={2017}
}
```

## BSN

[Bsn: Boundary sensitive network for temporal action proposal generation](https://openaccess.thecvf.com/content_ECCV_2018/html/Tianwei_Lin_BSN_Boundary_Sensitive_ECCV_2018_paper.html)

<!-- [ALGORITHM] -->

### Abstract

<!-- [ABSTRACT] -->

Temporal action proposal generation is an important yet challenging problem, since temporal proposals with rich action content are indispensable for analysing real-world videos with long duration and high proportion irrelevant content. This problem requires methods not only generating proposals with precise temporal boundaries, but also retrieving proposals to cover truth action instances with high recall and high overlap using relatively fewer proposals. To address these difficulties, we introduce an effective proposal generation method, named Boundary-Sensitive Network (BSN), which adopts "local to global" fashion. Locally, BSN first locates temporal boundaries with high probabilities, then directly combines these boundaries as proposals. Globally, with Boundary-Sensitive Proposal feature, BSN retrieves proposals by evaluating the confidence of whether a proposal contains an action within its region. We conduct experiments on two challenging datasets: ActivityNet-1.3 and THUMOS14, where BSN outperforms other state-of-the-art temporal action proposal generation methods with high recall and high temporal precision. Finally, further experiments demonstrate that by combining existing action classifiers, our method significantly improves the state-of-the-art temporal action detection performance.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/34324155/143016692-69efafbd-cec6-47f1-af45-371d0ff78a97.png" width="800"/>
</div>

### Results and Models

#### ActivityNet feature

|    feature    | gpus | pretrain |  AUC  | AR@1  | AR@5  | AR@10 | AR@100 |   gpu_mem(M)    | iter time(s) |                   config                   |                   ckpt                   |                   log                    |
| :-----------: | :--: | :------: | :---: | :---: | :---: | :---: | :----: | :-------------: | :----------: | :----------------------------------------: | :--------------------------------------: | :--------------------------------------: |
| cuhk_mean_100 |  1   |   None   | 66.26 | 32.71 | 48.43 | 55.28 | 74.27  | 43(TEM)+25(PEM) |      -       | [config_TEM](https://github.com/open-mmlab/mmaction2/tree/master/configs/localization/bsn/bsn_tem_1xb16-400x100-20e_activitynet-feature.py) [config_PGM](https://github.com/open-mmlab/mmaction2/tree/master/configs/localization/bsn/bsn_pgm_400x100_activitynet-feature.py) [config_PEM](https://github.com/open-mmlab/mmaction2/tree/master/configs/localization/bsn/bsn_pem_1xb16-400x100-20e_activitynet-feature.py) | [ckpt_TEM](https://download.openmmlab.com/mmaction/v1.0/localization/bsn/bsn_tem_1xb16-400x100-20e_activitynet-feature_20220908-9da79951.pth) [ckpt_PEM](https://download.openmmlab.com/mmaction/v1.0/localization/bsn/bsn_pem_1xb16-400x100-20e_activitynet-feature_20220908-ec2eb21d.pth) | [log_tem](https://download.openmmlab.com/mmaction/v1.0/localization/bsn/bsn_tem_1xb16-400x100-20e_activitynet-feature.log) [log_pem](https://download.openmmlab.com/mmaction/v1.0/localization/bsn/bsn_pem_1xb16-400x100-20e_activitynet-feature.log) |

1. The **gpus** indicates the number of gpu we used to get the checkpoint.
   According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you may set the learning rate proportional to the batch size if you use different GPUs or videos per GPU,
   e.g., lr=0.01 for 4 GPUs x 2 video/gpu and lr=0.08 for 16 GPUs x 4 video/gpu.
2. For feature column, cuhk_mean_100 denotes the widely used cuhk activitynet feature extracted by [anet2016-cuhk](https://github.com/yjxiong/anet2016-cuhk).

For more details on data preparation, you can refer to [ActivityNet Data Preparation](https://github.com/open-mmlab/mmaction2/tree/master/tools/data/activitynet/README.md).

### Training and Test

The traing of the BSN model is three-stages. Firstly train the Temporal evaluation module (TEM):

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

### Citation

```BibTeX
@inproceedings{lin2018bsn,
  title={Bsn: Boundary sensitive network for temporal action proposal generation},
  author={Lin, Tianwei and Zhao, Xu and Su, Haisheng and Wang, Chongjing and Yang, Ming},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={3--19},
  year={2018}
}
```
