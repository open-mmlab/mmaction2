# BMN

[Bmn: Boundary-matching network for temporal action proposal generation](https://openaccess.thecvf.com/content_ICCV_2019/html/Lin_BMN_Boundary-Matching_Network_for_Temporal_Action_Proposal_Generation_ICCV_2019_paper.html)

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

Temporal action proposal generation is an challenging and promising task which aims to locate temporal regions in real-world videos where action or event may occur. Current bottom-up proposal generation methods can generate proposals with precise boundary, but cannot efficiently generate adequately reliable confidence scores for retrieving proposals. To address these difficulties, we introduce the Boundary-Matching (BM) mechanism to evaluate confidence scores of densely distributed proposals, which denote a proposal as a matching pair of starting and ending boundaries and combine all densely distributed BM pairs into the BM confidence map. Based on BM mechanism, we propose an effective, efficient and end-to-end proposal generation method, named Boundary-Matching Network (BMN), which generates proposals with precise temporal boundaries as well as reliable confidence scores simultaneously. The two-branches of BMN are jointly trained in an unified framework. We conduct experiments on two challenging datasets: THUMOS-14 and ActivityNet-1.3, where BMN shows significant performance improvement with remarkable efficiency and generalizability. Further, combining with existing action classifier, BMN can achieve state-of-the-art temporal action detection performance.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/34324155/143016479-2ca7e8b6-a17b-4a4c-b4c9-ae731935cd91.png" width="800"/>
</div>

## Results and Models

### ActivityNet feature

|                                 config                                 |    feature    | gpus |  AUC  | AR@1  | AR@5  | AR@10 | AR@100 | gpu_mem(M) |                  ckpt                   |                  log                   |
| :--------------------------------------------------------------------: | :-----------: | :--: | :---: | :---: | :---: | :---: | :----: | :--------: | :-------------------------------------: | :------------------------------------: |
| [bmn_400x100_9e_2x8_activitynet_feature](/configs/localization/bmn/bmn_400x100_2x8_9e_activitynet_feature.py) | cuhk_mean_100 |  2   | 67.17 | 33.65 | 49.63 | 56.81 | 75.14  |    5420    | [ckpt](https://download.openmmlab.com/) | [log](https://download.openmmlab.com/) |
| [BMN-official](https://github.com/JJBOY/BMN-Boundary-Matching-Network) (for reference)\* | cuhk_mean_100 |  -   | 67.49 |   -   |   -   |   -   | 75.27  |     -      |                    -                    |                   -                    |

:::{note}

1. The **gpus** indicates the number of gpu we used to get the checkpoint.
   According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you may set the learning rate proportional to the batch size if you use different GPUs or videos per GPU,
   e.g., lr=0.01 for 4 GPUs x 2 video/gpu and lr=0.08 for 16 GPUs x 4 video/gpu.
2. For feature column, cuhk_mean_100 denotes the widely used cuhk activitynet feature extracted by [anet2016-cuhk](https://github.com/yjxiong/anet2016-cuhk).
3. We evaluate the action detection performance of BMN, using  [anet_cuhk_2017](https://download.openmmlab.com/mmaction/localization/cuhk_anet17_pred.json) submission for ActivityNet2017 Untrimmed Video Classification Track to assign label for each action proposal.

:::

\*We train BMN with the [official repo](https://github.com/JJBOY/BMN-Boundary-Matching-Network), evaluate its proposal generation and action detection performance with [anet_cuhk_2017](https://download.openmmlab.com/mmaction/localization/cuhk_anet17_pred.json) for label assigning.

For more details on data preparation, you can refer to ActivityNet feature in [Data Preparation](/docs/data_preparation.md).

## Train

Train BMN model on ActivityNet features dataset.

```shell
bash tools/dist_train.sh configs/localization/bmn/bmn_400x100_2x8_9e_activitynet_feature.py 2
```

For more details and optional arguments infos, you can refer to **Training setting** part in [getting_started](/docs/getting_started.md#training-setting) .

## Test

Test BMN on ActivityNet feature dataset.

```shell
python3 tools/test.py  configs/localization/bmn/bmn_400x100_2x8_9e_activitynet_feature.py CHECKPOINT.PTH
```

For more details and optional arguments infos, you can refer to **Test a dataset** part in [getting_started](/docs/getting_started.md#test-a-dataset) .

## Citation

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
