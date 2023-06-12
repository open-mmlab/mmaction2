# CLIP4Clip

[CLIP4Clip: An Empirical Study of CLIP for End to End Video Clip Retrieval](https://arxiv.org/abs/2104.08860)

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

Video-text retrieval plays an essential role in multi-modal research and has been widely used in many real-world web applications. The CLIP (Contrastive Language-Image Pre-training), an image-language pre-training model, has demonstrated the power of visual concepts learning from web collected image-text datasets. In this paper, we propose a CLIP4Clip model to transfer the knowledge of the CLIP model to video-language retrieval in an end-to-end manner. Several questions are investigated via empirical studies: 1) Whether image feature is enough for video-text retrieval? 2) How a post-pretraining on a large-scale video-text dataset based on the CLIP affect the performance? 3) What is the practical mechanism to model temporal dependency between video frames? And 4) The Hyper-parameters sensitivity of the model on video-text retrieval task. Extensive experimental results present that the CLIP4Clip model transferred from the CLIP can achieve SOTA results on various video-text retrieval datasets, including MSR-VTT, MSVC, LSMDC, ActivityNet, and DiDeMo.

<!-- [IMAGE] -->

<div align=center>
<img src="https://github.com/Dai-Wenxun/mmaction2/assets/58767402/f91fc927-d5f2-41dd-8198-def71d392991" width="800"/>
</div>

## Results and Models

### MSRVTT-9k

| frame sampling strategy |   resolution   | gpus |  backbone   | adapter | Recall@1 | Recall@5 | Recall@10 | MdR | MnR |[reference](<(https://github.com/Sense-X/UniFormer/blob/main/video_classification/README.md)>) top1 acc | [reference](<(https://github.com/Sense-X/UniFormer/blob/main/video_classification/README.md)>) top5 acc | mm-Kinetics top1 acc | mm-Kinetics top5 acc | testing protocol | FLOPs | params |                                              config                                               |                                                                           ckpt                                                                           |
| :---------------------: | :------------: |  :---------: | :---------: | :------: |:------: | :------: | :-----------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------: | :------------------: | :------------------: | :--------------: | :---: | :----: | :-----------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------: |
|         uniform 12    | 224x224 | 8 | CLIP | Mean |    |   94.6   |                                                  80.8                                                   |                                                  94.7                                                   |         80.9         |         94.6         | 4 clips x 1 crop | 41.8G | 21.4M  | [config](/configs/recognition/uniformer/uniformer-small_imagenet1k-pre_16x4x1_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/uniformerv1/uniformer-small_imagenet1k-pre_16x4x1_kinetics400-rgb_20221219-c630a037.pth) |

1. The **gpus** indicates the number of gpus we used to get the checkpoint. It is noteworthy that the configs we provide are used for 8 gpus as default.
   According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you may set the learning rate proportional to the batch size if you use different GPUs or videos per GPU,
   e.g., lr=0.01 for 4 GPUs x 2 video/gpu and lr=0.08 for 16 GPUs x 4 video/gpu.

For more details on data preparation, you can refer to `Prepare audio` in [Data Preparation Tutorial](/docs/en/user_guides/2_data_prepare.md).

## Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train CLIP4Clip model on MSRVTT-9k dataset in a deterministic option with periodic validation.

```shell
python tools/train.py configs/retrieval/clip4clip/clip4clip_vit-base-p32-res224-clip-pre_8xb16-u12-5e_msrvtt-9k-rgb.py \
    --cfg-options randomness.seed=0 randomness.deterministic=True
```

For more details, you can refer to the **Training** part in the [Training and Test Tutorial](/docs/en/user_guides/train_test.md).

## Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test CLIP4Clip model on MSRVTT-9k dataset and dump the result to a pkl file.

```shell
python tools/test.py configs/retrieval/clip4clip/clip4clip_vit-base-p32-res224-clip-pre_8xb16-u12-5e_msrvtt-9k-rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --dump result.pkl
```

For more details, you can refer to the **Test** part in the [Training and Test Tutorial](/docs/en/user_guides/train_test.md).

## Citation

```BibTeX
@article{luo2022clip4clip,
  title={CLIP4Clip: An empirical study of CLIP for end to end video clip retrieval and captioning},
  author={Luo, Huaishao and Ji, Lei and Zhong, Ming and Chen, Yang and Lei, Wen and Duan, Nan and Li, Tianrui},
  journal={Neurocomputing},
  volume={508},
  pages={293--304},
  year={2022},
}
```
