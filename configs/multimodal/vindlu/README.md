# VindLU

[VindLU: A Recipe for Effective Video-and-Language Pretraining](https://arxiv.org/abs/2212.05051)

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

The last several years have witnessed remarkable progress in video-and-language (VidL) understanding. However, most modern VidL approaches use complex and specialized model architectures and sophisticated pretraining protocols, making the reproducibility, analysis and comparisons of these frameworks difficult. Hence, instead of proposing yet another new VidL model, this paper conducts a thorough empirical study demystifying the most important factors in the VidL model design. Among the factors that we investigate are (i) the spatiotemporal architecture design, (ii) the multimodal fusion schemes, (iii) the pretraining objectives, (iv) the choice of pretraining data, (v) pretraining and finetuning protocols, and (vi) dataset and model scaling. Our empirical study reveals that the most important design factors include: temporal modeling, video-to-text multimodal fusion, masked modeling objectives, and joint training on images and videos. Using these empirical insights, we then develop a step-by-step recipe, dubbed VindLU, for effective VidL pretraining. Our final model trained using our recipe achieves comparable or better than state-of-the-art results on several VidL tasks without relying on external CLIP pretraining. In particular, on the text-to-video retrieval task, our approach obtains 61.2% on DiDeMo, and 55.0% on ActivityNet, outperforming current SOTA by 7.8% and 6.1% respectively. Furthermore, our model also obtains state-of-the-art video question-answering results on ActivityNet-QA, MSRVTT-QA, MSRVTT-MC and TVQA. Our code and pretrained models are publicly available at: https://github.com/klauscc/VindLU.

<!-- [IMAGE] -->

<div align=center>
<img src="https://github.com/open-mmlab/mmaction2/assets/33249023/3efb02d0-679f-4ce7-b8f1-7f331905d902" width="800"/>
</div>

## Results and Models

### Video Retrieval on MSRVTT-9k

| frame sampling strategy | resolution | gpus | vision encoder | text encoder |      pretraining       | Recall@1 |                config                 |                ckpt                 |                 log                 |
| :---------------------: | :--------: | :--: | :------------: | :----------: | :--------------------: | :------: | :-----------------------------------: | :---------------------------------: | :---------------------------------: |
|       uniform 12        |  224x224   |  8   |   BEiT-Base    |  Bert-Base   | C5M (WebVid-2M + CC3M) |   44.0   | [config](/configs/multimodal/vindlu/vindlu_beit-base_8x16_retrieval_msrvtt-9k.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/multimodal/vindlu/vindlu_beit-base_8x16_retrieval_msrvtt-9k/vindlu_beit-base_8x16_retrieval_msrvtt-9k_20230905-fc36231e.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/multimodal/vindlu/vindlu_beit-base_8x16_retrieval_msrvtt-9k/vindlu_beit-base_8x16_retrieval_msrvtt-9k.log) |

### Video Question-Answering on MSRVTT-QA

| frame sampling strategy | resolution | gpus | vision encoder | text encoder |      pretraining       | top1 acc |                config                 |                ckpt                 |                 log                 |
| :---------------------: | :--------: | :--: | :------------: | :----------: | :--------------------: | :------: | :-----------------------------------: | :---------------------------------: | :---------------------------------: |
|       uniform 12        |  224x224   |  8   |   BEiT-Base    |  Bert-Base   | C5M (WebVid-2M + CC3M) |   43.6   | [config](/configs/multimodal/vindlu/vindlu_beit-base_8x8_vqa_msrvtt-qa.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/multimodal/vindlu/vindlu_beit-base_8x8_vqa_msrvtt-qa/vindlu_beit-base_8x8_vqa_msrvtt-qa_20230906-6e693e64.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/multimodal/vindlu/vindlu_beit-base_8x8_vqa_msrvtt-qa/vindlu_beit-base_8x8_vqa_msrvtt-qa.log) |

### Multiple-Choice Question-Answering on MSRVTT-MC (Inference)

| frame sampling strategy | resolution | gpus | vision encoder | text encoder |      pretraining       | top1 acc |                         config                         |                         ckpt                          |
| :---------------------: | :--------: | :--: | :------------: | :----------: | :--------------------: | :------: | :----------------------------------------------------: | :---------------------------------------------------: |
|       uniform 12        |  224x224   |  8   |   BEiT-Base    |  Bert-Base   | C5M (WebVid-2M + CC3M) |   97.6   | [config](/configs/multimodal/vindlu/vindlu_beit-base_vqa-mc_msrvtt-mc.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/multimodal/vindlu/vindlu_beit-base_8x16_retrieval_msrvtt-9k/vindlu_beit-base_8x16_retrieval_msrvtt-9k_20230905-fc36231e.pth) |

1. Currently, we only support the fine-tuning stage of VindLU models based on the pretrained checkpoint provided by the [original repo](https://github.com/klauscc/VindLU).

For more details on data preparation, you can refer to [prepare msrvtt](/tools/data/msrvtt/README.md).

## Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train VindLU model on MSRVTT-9k dataset in a deterministic option with periodic validation.

```shell
python tools/train.py configs/multimodal/vindlu/vindlu_beit-base_8x16_retrieval_msrvtt-9k.py \
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
python tools/test.py cconfigs/multimodal/vindlu/vindlu_beit-base_8x16_retrieval_msrvtt-9k.py \
    checkpoints/SOME_CHECKPOINT.pth --dump result.pkl
```

For more details, you can refer to the **Test** part in the [Training and Test Tutorial](/docs/en/user_guides/train_test.md).

## Citation

```BibTeX
@inproceedings{cheng2023vindlu,
  title={Vindlu: A recipe for effective video-and-language pretraining},
  author={Cheng, Feng and Wang, Xizi and Lei, Jie and Crandall, David and Bansal, Mohit and Bertasius, Gedas},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={10739--10750},
  year={2023}
}
```
