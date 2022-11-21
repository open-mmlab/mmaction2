# EVL

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

Video recognition has been dominated by the end-to-end learning paradigm -- first initializing a video recognition model with weights of a pretrained image model and then conducting end-to-end training on videos. This enables the video network to benefit from the pretrained image model. However, this requires substantial computation and memory resources for finetuning on videos and the alternative of directly using pretrained image features without finetuning the image backbone leads to subpar results. Fortunately, recent advances in Contrastive Vision-Language Pre-training (CLIP) pave the way for a new route for visual recognition tasks. Pretrained on large open-vocabulary image-text pair data, these models learn powerful visual representations with rich semantics. In this paper, we present Efficient Video Learning (EVL) -- an efficient framework for directly training high-quality video recognition models with frozen CLIP features. Specifically, we employ a lightweight Transformer decoder and learn a query token to dynamically collect frame-level spatial features from the CLIP image encoder. Furthermore, we adopt a local temporal module in each decoder layer to discover temporal clues from adjacent frames and their attention maps. We show that despite being efficient to train with a frozen backbone, our models learn high quality video representations on a variety of video recognition datasets.

<!-- [IMAGE] -->

<div align=center>
<img src="TODO" width="800"/>

</div>

## Results and Models

### Kinetics-400

| frame sampling strategy | scheduler |   resolution   | gpus | backbone | pretrain | top1 acc | top5 acc |   reference top1 acc    |   reference top5 acc    | testing protocol  | gpu_mem(M) | params(M) | Flops(G) |   config    |   ckpt    |   log    |
| :---------------------: | :-------: | :------------: | :--: | :------: | :------: | :------: | :------: | :---------------------: | :---------------------: | :---------------: | :--------: | :-------: | :------: | :---------: | :-------: | :------: |
|         8x16x1          | MultiStep | short-side 320 |  8   | ViT-B/16 |   CLIP   |  82.17   |  95.63   | 82.6(test_in_mmaction2) | 96.0(test_in_mmaction2) | 10 clips x 3 crop |    TODO    |   TODO    |   TODO   | [config](/configs/recognition/evl/xxx.py) | [TODO](TODO) | [TODO](TODO) |

### SSv2

| frame sampling strategy | scheduler |   resolution   | gpus | backbone | pretrain | top1 acc | top5 acc |  reference top1 acc   | reference top5 acc | testing protocol  | gpu_mem(M) | params(M) | Flops(G) |    config     |    ckpt     |     log     |
| :---------------------: | :-------: | :------------: | :--: | :------: | :------: | :------: | :------: | :-------------------: | :----------------: | :---------------: | :--------: | :-------: | :------: | :-----------: | :---------: | :---------: |
|         8x16x1          | MultiStep | short-side 320 |  8   | ViT-B/16 |   CLIP   |  59.10   |  84.62   | 58.6(copy_from_paper) |        N/A         | 10 clips x 3 crop |    TODO    |   TODO    |   TODO   | [config](/configs/recognition/evl/xxx.py) | [TODO](TODO) | [TODO](TODO) |
