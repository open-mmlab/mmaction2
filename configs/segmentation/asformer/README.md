# ASFormer

[ASFormer: Transformer for Action Segmentation](https://arxiv.org/pdf/2110.08568.pdf)

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

Algorithms for the action segmentation task typically use temporal models to predict
what action is occurring at each frame for a minute-long daily activity. Recent studies have shown the potential of Transformer in modeling the relations among elements
in sequential data. However, there are several major concerns when directly applying
the Transformer to the action segmentation task, such as the lack of inductive biases
with small training sets, the deficit in processing long input sequence, and the limitation of the decoder architecture to utilize temporal relations among multiple action segments to refine the initial predictions. To address these concerns, we design an efficient
Transformer-based model for the action segmentation task, named ASFormer, with three
distinctive characteristics: (i) We explicitly bring in the local connectivity inductive priors because of the high locality of features. It constrains the hypothesis space within a
reliable scope, and is beneficial for the action segmentation task to learn a proper target
function with small training sets. (ii) We apply a pre-defined hierarchical representation pattern that efficiently handles long input sequences. (iii) We carefully design the
decoder to refine the initial predictions from the encoder. Extensive experiments on
three public datasets demonstrate the effectiveness of our methods. The original code is available at
https://github.com/ChinaYi/ASFormer.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/34324155/143016479-2ca7e8b6-a17b-4a4c-b4c9-ae731935cd91.png" width="800"/>
</div>

## Results and Models

### ActivityNet feature

| feature | gpus | pretrain |  ACC  | EDIT  | F1@10 | F1@25 | F1@50 | gpu_mem(M) | iter time(s) |                     config                     |                     ckpt                     |                     log                      |
| :-----: | :--: | :------: | :---: | :---: | :---: | :---: | :---: | :--------: | :----------: | :--------------------------------------------: | :------------------------------------------: | :------------------------------------------: |
|  gtea   |  1   |   None   | 67.25 | 32.89 | 49.43 | 56.64 | 75.29 |    8693    |      -       | [config](/configs/localization/bmn/bmn_2xb8-400x100-9e_activitynet-feature.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/localization/bmn/bmn_2xb8-400x100-9e_activitynet-feature_20220908-79f92857.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/localization/bmn/bmn_2xb8-400x100-9e_activitynet-feature.log) |

1. The **gpus** indicates the number of gpu we used to get the checkpoint.
   According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you may set the learning rate proportional to the batch size if you use different GPUs or videos per GPU,
   e.g., lr=0.01 for 4 GPUs x 2 video/gpu and lr=0.08 for 16 GPUs x 4 video/gpu.
2. For feature column, cuhk_mean_100 denotes the widely used cuhk activitynet feature extracted by [anet2016-cuhk](https://github.com/yjxiong/anet2016-cuhk).
3. We evaluate the action detection performance of BMN, using  [anet_cuhk_2017](https://download.openmmlab.com/mmaction/localization/cuhk_anet17_pred.json) submission for ActivityNet2017 Untrimmed Video Classification Track to assign label for each action proposal.

\*We train BMN with the [official repo](https://github.com/JJBOY/BMN-Boundary-Matching-Network), evaluate its proposal generation and action detection performance with [anet_cuhk_2017](https://download.openmmlab.com/mmaction/localization/cuhk_anet17_pred.json) for label assigning.

For more details on data preparation, you can refer to [ActivityNet Data Preparation](/tools/data/activitynet/README.md).

## Train

Train ASFormer model on features dataset for action segmentation.

```shell
bash tools/dist_train.sh configs/segmentation/asformer/asformer_gtea.py 1
```

For more details, you can refer to the **Training** part in the [Training and Test Tutorial](/docs/en/user_guides/train_test.md).

## Test

Test ASFormer on features dataset for action segmentation.

```shell
python3 tools/test.py  configs/segmentation/asformer/asformer_gtea.py CHECKPOINT.PTH
```

For more details, you can refer to the **Testing** part in the [Training and Test Tutorial](/docs/en/user_guides/train_test.md).

## Citation

```BibTeX
@inproceedings{chinayi_ASformer,
	author={Fangqiu Yi and Hongyu Wen and Tingting Jiang},
	booktitle={The British Machine Vision Conference (BMVC)},
	title={ASFormer: Transformer for Action Segmentation},
	year={2021},
}
```

<!-- [DATASET] -->

```BibTeX
@inproceedings{fathi2011learning,
  title={Learning to recognize objects in egocentric activities},
  author={Fathi, Alireza and Ren, Xiaofeng and Rehg, James M},
  booktitle={CVPR 2011},
  pages={3281--3288},
  year={2011},
  organization={IEEE}
}
```
