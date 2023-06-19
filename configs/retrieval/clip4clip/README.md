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

| frame sampling strategy | resolution | gpus | backbone | adapter | pretrain | Recall@1 | Recall@5 | Recall@10 | MdR | MnR  | testing protocol |              config              |              ckpt              |              log               |
| :---------------------: | :--------: | :--: | :------: | :-----: | :------: | :------: | :------: | :-------: | :-: | :--: | :--------------: | :------------------------------: | :----------------------------: | :----------------------------: |
|       uniform 12        |  224x224   |  8   | ViT-B/32 |  Mean   |   clip   |   43.1   |   69.4   |   78.9    | 2.0 | 16.8 | 1 clips x 1 crop | [config](/configs/retrieval/clip4clip/clip4clip_vit-base-p32-res224-clip-pre_8xb16-u12-5e_msrvtt-9k-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/retrieval/clip4clip/clip4clip_vit-base-p32-res224-clip-pre_8xb16-u12-5e_msrvtt-9k-rgb/clip4clip_vit-base-p32-res224-clip-pre_8xb16-u12-5e_msrvtt-9k-rgb_20230612-b9706e54.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/retrieval/clip4clip/clip4clip_vit-base-p32-res224-clip-pre_8xb16-u12-5e_msrvtt-9k-rgb/clip4clip_vit-base-p32-res224-clip-pre_8xb16-u12-5e_msrvtt-9k-rgb.log) |

For more details on data preparation, you can refer to [video_retrieval](/tools/data/video_retrieval/README.md).

## Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train CLIP4Clip model on MSRVTT-9k dataset in a deterministic option with periodic validation.

```shell
python tools/train.py configs/retrieval/clip4clip/clip4clip_vit-base-p32-res224-clip-pre_8xb16-u12-5e_msrvtt-9k-rgb.py \
    --seed 0 --deterministic
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
