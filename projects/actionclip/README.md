# ActionCLIP Project

[ActionCLIP: A New Paradigm for Video Action Recognition](https://arxiv.org/abs/2109.08472)

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

The canonical approach to video action recognition dictates a neural model to do a classic and standard 1-of-N majority vote task. They are trained to predict a fixed set of predefined categories, limiting their transferable ability on new datasets with unseen concepts. In this paper, we provide a new perspective on action recognition by attaching importance to the semantic information of label texts rather than simply mapping them into numbers. Specifically, we model this task as a video-text matching problem within a multimodal learning framework, which strengthens the video representation with more semantic language supervision and enables our model to do zero-shot action recognition without any further labeled data or parameters requirements. Moreover, to handle the deficiency of label texts and make use of tremendous web data, we propose a new paradigm based on this multimodal learning framework for action recognition, which we dub "pre-train, prompt and fine-tune". This paradigm first learns powerful representations from pre-training on a large amount of web image-text or video-text data. Then it makes the action recognition task to act more like pre-training problems via prompt engineering. Finally, it end-to-end fine-tunes on target datasets to obtain strong performance. We give an instantiation of the new paradigm, ActionCLIP, which not only has superior and flexible zero-shot/few-shot transfer ability but also reaches a top performance on general action recognition task, achieving 83.8% top-1 accuracy on Kinetics-400 with a ViT-B/16 as the backbone.

<!-- [IMAGE] -->

<div align=center>
<img src="https://github-production-user-asset-6210df.s3.amazonaws.com/58767402/237413093-75d76018-0521-4642-af68-32141fb4fed1.png" width="800"/>
</div>

## Usage

### Setup Environment

Please refer to [Installation](https://mmaction2.readthedocs.io/en/latest/get_started/installation.html) to install MMAction2. Run the following command to install `clip`.

```shell
pip install git+https://github.com/openai/CLIP.git
```

Assume that you are located at `$MMACTION2/projects/actionclip`.

Add the current folder to `PYTHONPATH`, so that Python can find your code. Run the following command in the current directory to add it.

> Please run it every time after you opened a new shell.

```shell
export PYTHONPATH=`pwd`:$PYTHONPATH
```

### Data Preparation

Prepare the Kinetics400 dataset according to the [instruction](https://github.com/open-mmlab/mmaction2/blob/main/tools/data/kinetics/README.md).

Create a symbolic link from `$MMACTION2/data` to `./data` in the current directory, so that Python can locate your data. Run the following command in the current directory to create the symbolic link.

```shell
ln -s ../../data ./data
```

### Training commands

**To train with single GPU:**

```bash
mim train mmaction configs/actionclip_vit-base-p32-res224-clip-pre_g8xb16_1x1x8_k400-rgb.py
```

**To train with multiple GPUs:**

```bash
mim train mmaction configs/actionclip_vit-base-p32-res224-clip-pre_g8xb16_1x1x8_k400-rgb.py --launcher pytorch --gpus 8
```

**To train with multiple GPUs by slurm:**

```bash
mim train mmaction configs/actionclip_vit-base-p32-res224-clip-pre_g8xb16_1x1x8_k400-rgb.py --launcher slurm \
    --gpus 8 --gpus-per-node 8 --partition $PARTITION
```

### Testing commands

**To test with single GPU:**

```bash
mim test mmaction configs/actionclip_vit-base-p32-res224-clip-pre_g8xb16_1x1x8_k400-rgb.py --checkpoint $CHECKPOINT
```

**To test with multiple GPUs:**

```bash
mim test mmaction configs/actionclip_vit-base-p32-res224-clip-pre_g8xb16_1x1x8_k400-rgb.py --checkpoint $CHECKPOINT --launcher pytorch --gpus 8
```

**To test with multiple GPUs by slurm:**

```bash
mim test mmaction configs/actionclip_vit-base-p32-res224-clip-pre_g8xb16_1x1x8_k400-rgb.py --checkpoint $CHECKPOINT --launcher slurm \
    --gpus 8 --gpus-per-node 8 --partition $PARTITION
```

## Results

### Kinetics400

| frame sampling strategy | backbone | top1 acc | top5 acc |  testing protocol  |                                config                                |                                ckpt                                 |
| :---------------------: | :------: | :------: | :------: | :----------------: | :------------------------------------------------------------------: | :-----------------------------------------------------------------: |
|          1x1x8          | ViT-B/32 |   77.6   |   93.8   | 8 clips  x 1 crop  | [config](./configs/actionclip_vit-base-p32-res224-clip-pre_1x1x8_k400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/projects/actionclip/actionclip_vit-base-p32-res224-clip-pre_1x1x8_k400-rgb/vit-b-32-8f.pth)\[1\] |
|          1x1x8          | ViT-B/16 |   80.3   |   95.2   | 8 clips  x 1 crop  | [config](./configs/actionclip_vit-base-p16-res224-clip-pre_1x1x8_k400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/projects/actionclip/actionclip_vit-base-p16-res224-clip-pre_1x1x8_k400-rgb/vit-b-16-8f.pth)\[1\] |
|         1x1x16          | ViT-B/16 |   81.1   |   95.6   | 16 clips  x 1 crop | [config](./configs/actionclip_vit-base-p16-res224-clip-pre_1x1x16_k400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/projects/actionclip/actionclip_vit-base-p16-res224-clip-pre_1x1x16_k400-rgb/vit-b-16-16f.pth)\[1\] |
|         1x1x32          | ViT-B/16 |   81.3   |   95.8   | 32 clips  x 1 crop | [config](./configs/actionclip_vit-base-p16-res224-clip-pre_1x1x32_k400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/projects/actionclip/actionclip_vit-base-p16-res224-clip-pre_1x1x32_k400-rgb/vit-b-16-32f.pth)\[1\] |

\[1\] The models are ported from the repo [ActionCLIP](https://github.com/sallymmx/ActionCLIP) and tested on our data. Currently, we only support the testing of ActionCLIP models. Due to the variation in testing data, our reported test accuracy differs from that of the original repository (on average, it is lower by one point). Please refer to this [issue](https://github.com/sallymmx/ActionCLIP/issues/14) for more details.

### Kinetics400 (Trained on Our K400 dataset)

| frame sampling strategy | gpus | backbone | top1 acc | top5 acc | testing protocol  |                    config                     |                     ckpt                     |                     log                     |
| :---------------------: | :--: | :------: | :------: | :------: | :---------------: | :-------------------------------------------: | :------------------------------------------: | :-----------------------------------------: |
|          1x1x8          |  8   | ViT-B/32 |   77.5   |   93.2   | 8 clips  x 1 crop | [config](./configs/actionclip_vit-base-p32-res224-clip-pre_g8xb16_1x1x8_k400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/projects/actionclip/actionclip_vit-base-p32-res224-clip-pre_g8xb16_1x1x8_k400-rgb/actionclip_vit-base-p32-res224-clip-pre_g8xb16_1x1x8_k400-rgb_20230801-8535b794.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/projects/actionclip/actionclip_vit-base-p32-res224-clip-pre_g8xb16_1x1x8_k400-rgb/actionclip_vit-base-p32-res224-clip-pre_g8xb16_1x1x8_k400-rgb.log) |
|          1x1x8          |  8   | ViT-B/16 |   81.3   |   95.2   | 8 clips  x 1 crop | [config](./configs/actionclip_vit-base-p16-res224-clip-pre_g8xb16_1x1x8_k400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/projects/actionclip/actionclip_vit-base-p16-res224-clip-pre_g8xb16_1x1x8_k400-rgb/actionclip_vit-base-p16-res224-clip-pre_g8xb16_1x1x8_k400-rgb_20230801-b307a0cd.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/projects/actionclip/actionclip_vit-base-p16-res224-clip-pre_g8xb16_1x1x8_k400-rgb/actionclip_vit-base-p16-res224-clip-pre_g8xb16_1x1x8_k400-rgb.log) |

## Zero-Shot Prediction

We offer two methods for zero-shot prediction as follows. The `test.mp4` can be downloaded from [here](https://github-production-user-asset-6210df.s3.amazonaws.com/58767402/237333525-89ebee9a-573e-4e27-9047-0ad6422fa82f.mp4).

### Using Naive Pytorch

```python
import torch
import clip
from models.load import init_actionclip
from mmaction.utils import register_all_modules

register_all_modules(True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = init_actionclip('ViT-B/32-8', device=device)

video_anno = dict(filename='test.mp4', start_index=0)
video = preprocess(video_anno).unsqueeze(0).to(device)

template = 'The woman is {}'
labels = ['singing', 'dancing', 'performing']
text = clip.tokenize([template.format(label) for label in labels]).to(device)

with torch.no_grad():
    video_features = model.encode_video(video)
    text_features = model.encode_text(text)

video_features /= video_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = (100 * video_features @ text_features.T).softmax(dim=-1)
probs = similarity.cpu().numpy()

print("Label probs:", probs)  # [[9.995e-01 5.364e-07 6.666e-04]]
```

### Using MMAction2 APIs

```python
import mmengine
import torch
from mmaction.utils import register_all_modules
from mmaction.apis import inference_recognizer, init_recognizer

register_all_modules(True)

config_path = 'configs/actionclip_vit-base-p32-res224-clip-pre_1x1x8_k400-rgb.py'
checkpoint_path = 'https://download.openmmlab.com/mmaction/v1.0/projects/actionclip/actionclip_vit-base-p32-res224-clip-pre_1x1x8_k400-rgb/vit-b-32-8f.pth'
template = 'The woman is {}'
labels = ['singing', 'dancing', 'performing']

# Update the labels, the default is the label list of K400.
config = mmengine.Config.fromfile(config_path)
config.model.labels_or_label_file = labels
config.model.template = template

device = "cuda" if torch.cuda.is_available() else "cpu"
model = init_recognizer(config=config, checkpoint=checkpoint_path, device=device)

pred_result = inference_recognizer(model, 'test.mp4')
probs = pred_result.pred_score.cpu().numpy()
print("Label probs:", probs)  # [9.995e-01 5.364e-07 6.666e-04]
```

## Citation

<!-- Replace to the citation of the paper your project refers to. -->

```bibtex
@article{wang2021actionclip,
  title={Actionclip: A new paradigm for video action recognition},
  author={Wang, Mengmeng and Xing, Jiazheng and Liu, Yong},
  journal={arXiv preprint arXiv:2109.08472},
  year={2021}
}
```
