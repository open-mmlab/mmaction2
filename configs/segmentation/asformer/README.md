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
three public datasets demonstrate the effectiveness of our methods.

<!-- [IMAGE] -->

<div align=center>
<img src="https://github.com/open-mmlab/mmaction2/assets/35267818/ea2af27e-0cd9-489d-9c81-02b8a7f29ef1" width="800"/>
</div>

## Results

### GTEA

| split  | gpus | pretrain |  ACC  | EDIT  | F1@10 | F1@25 | F1@50 | gpu_mem(M) |                       config                       |                       ckpt                        |                       log                        |
| :----: | :--: | :------: | :---: | :---: | :---: | :---: | :---: | :--------: | :------------------------------------------------: | :-----------------------------------------------: | :----------------------------------------------: |
| split2 |  1   |   None   | 80.34 | 81.58 | 89.30 | 87.83 | 75.28 |    1500    | [config](/configs/segmentation/asformer/asformer_1xb1-120e_gtea-split2-i3d-feature.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/segmentation/asformer/asformer_1xb1-120e_gtea-split2-i3d-feature_20231011-b5aaf789.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/segmentation/asformer/asformer_1xb1-120e_gtea-split2-i3d-feature.log) |
| split1 |  1   |   None   | 76.54 | 80.36 | 84.80 | 83.39 | 77.74 |    1500    |                         -                          |                         -                         |                        -                         |
| split3 |  1   |   None   | 82.41 | 90.03 | 92.13 | 92.37 | 86.26 |    1500    |                         -                          |                         -                         |                        -                         |
| split4 |  1   |   None   | 79.77 | 91.70 | 92.88 | 92.39 | 81.65 |    1500    |                         -                          |                         -                         |                        -                         |

### 50Salads

| split  | gpus | pretrain |  ACC  | EDIT  | F1@10 | F1@25 | F1@50 | gpu_mem(M) |                       config                       |                       ckpt                        |                       log                        |
| :----: | :--: | :------: | :---: | :---: | :---: | :---: | :---: | :--------: | :------------------------------------------------: | :-----------------------------------------------: | :----------------------------------------------: |
| split2 |  1   |   None   | 87.55 | 79.10 | 85.17 | 83.73 | 77.99 |    7200    | [config](/configs/segmentation/asformer/asformer_1xb1-120e_50salads-split2-i3d-feature.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/segmentation/asformer/asformer_1xb1-120e_50salads-split2-i3d-feature_20231011-25dc57d5.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/segmentation/asformer/asformer_1xb1-120e_50salads-split2-i3d-feature.log) |
| split1 |  1   |   None   | 81.44 | 73.25 | 82.04 | 80.27 | 71.84 |    7200    |                         -                          |                         -                         |                        -                         |
| split3 |  1   |   None   | 85.51 | 82.23 | 85.71 | 84.29 | 78.57 |    7200    |                         -                          |                         -                         |                        -                         |
| split4 |  1   |   None   | 87.27 | 80.46 | 85.99 | 83.14 | 78.86 |    7200    |                         -                          |                         -                         |                        -                         |
| split5 |  1   |   None   | 87.96 | 75.29 | 84.60 | 83.13 | 76.28 |    7200    |                         -                          |                         -                         |                        -                         |

### Breakfast

| split  | gpus | pretrain |  ACC  | EDIT  | F1@10 | F1@25 | F1@50 | gpu_mem(M) |                       config                       |                       ckpt                        |                       log                        |
| :----: | :--: | :------: | :---: | :---: | :---: | :---: | :---: | :--------: | :------------------------------------------------: | :-----------------------------------------------: | :----------------------------------------------: |
| split2 |  1   |   None   | 74.12 | 76.53 | 77.74 | 72.62 | 60.43 |    8800    | [config](/configs/segmentation/asformer/asformer_1xb1-120e_breakfast-split2-i3d-feature.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/segmentation/asformer/asformer_1xb1-120e_breakfast-split2-i3d-feature_20231011-10e557f3.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/segmentation/asformer/asformer_1xb1-120e_breakfast-split2-i3d-feature.log) |
| split1 |  1   |   None   | 75.52 | 76.87 | 77.06 | 73.05 | 61.77 |    8800    |                         -                          |                         -                         |                        -                         |
| split3 |  1   |   None   | 74.86 | 74.33 | 76.17 | 70.85 | 58.07 |    8800    |                         -                          |                         -                         |                        -                         |
| split4 |  1   |   None   | 70.39 | 71.54 | 73.42 | 66.61 | 52.76 |    8800    |                         -                          |                         -                         |                        -                         |

1. The **gpus** indicates the number of gpu we used to get the checkpoint.

2. We only provide checkpoints of one split. For experiments with other splits, we simply change the names of the training and testing datasets in the configs file, i.e., modifying `ann_file_train`, `ann_file_val` and `ann_file_test`.

For more details on data preparation, you can refer to [Preparing Datasets for Action Segmentation](/tools/data/action_seg/README.md).

## Train

Train ASFormer model on features dataset for action segmentation.

```shell
bash tools/dist_train.sh configs/segmentation/asformer/asformer_1xb1-120e_gtea-split2-i3d-feature.py 1
```

For more details, you can refer to the **Training** part in the [Training and Test Tutorial](/docs/en/user_guides/train_test.md).

## Test

Test ASFormer on features dataset for action segmentation.

```shell
python3 tools/test.py  configs/segmentation/asformer/asformer_1xb1-120e_gtea-split2-i3d-feature.py CHECKPOINT.PTH
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

```BibTeX
@inproceedings{stein2013combining,
  title={Combining embedded accelerometers with computer vision for recognizing food preparation activities},
  author={Stein, Sebastian and McKenna, Stephen J},
  booktitle={Proceedings of the 2013 ACM international joint conference on Pervasive and ubiquitous computing},
  pages={729--738},
  year={2013}
}
```

```BibTeX
@inproceedings{kuehne2014language,
  title={The language of actions: Recovering the syntax and semantics of goal-directed human activities},
  author={Kuehne, Hilde and Arslan, Ali and Serre, Thomas},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={780--787},
  year={2014}
}
```
