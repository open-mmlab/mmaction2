# A 20-Minute Guide to MMAction2 FrameWork

Next, we will demonstrate the overall architecture of our `MMACTION2 1.0` through a step-by-step example of video action recognition.

The structure of this tutorial is as follows:

- [A 20-Minute Guide to MMAction2 FrameWork](#a-20-minute-guide-to-mmaction2-framework)
  - [Step0: Prepare Data](#step0-prepare-data)
  - [Step1: Build a Pipeline](#step1-build-a-pipeline)
  - [Step2: Build a Dataset and DataLoader](#step2-build-a-dataset-and-dataloader)


## Step0: Prepare Data

Please download our self-made [kinetics400_tiny](https://download.openmmlab.com/mmaction/kinetics400_tiny.zip) dataset and extract it to the `$MMACTION2/data` directory.
The directory structure after extraction should be as follows:

```
mmaction2
├── data
│   ├── kinetics400_tiny
│   │    ├── kinetics_tiny_train_video.txt
│   │    ├── kinetics_tiny_val_video.txt
│   │    ├── train
│   │    │   ├── 27_CSXByd3s.mp4
│   │    │   ├── 34XczvTaRiI.mp4
│   │    │   ├── A-wiliK50Zw.mp4
│   │    │   ├── ... 
│   │    └── val
│   │       ├── 0pVGiAU6XEA.mp4
│   │       ├── AQrbRSnRt8M.mp4
│   │       ├── ...
```

Here are some examples from the annotation file `kinetics_tiny_train_video.txt`:

```
D32_1gwq35E.mp4 0
iRuyZSKhHRg.mp4 1
oXy-e_P_cAI.mp4 0
34XczvTaRiI.mp4 1
h2YqqUhnR34.mp4 0
```

Each line in the file represents the annotation of a video, where the first item denotes the video filename (e.g., `D32_1gwq35E.mp4`), and the second item represents the corresponding label (e.g., label `0` for `D32_1gwq35E.mp4`). In this dataset, there are only two categories.

## Step1: Build a Pipeline

In order to `decode`, `sample`, `resize`, `crop`, `format`, and `pack` the input video and corresponding annotation, we need to design a pipeline to handle these processes. Specifically, we design 7 `Transform` classes to build this video processing pipeline. Please note that all `Transform` classes in OpenMMLab must inherit from the `BaseTransform` class in `mmcv`, implement the abstract method `transform`, and be registered to the `TRANSFORMS` registry. For more detailed information about registry, please refer to [MMEngine Tutorial](https://mmengine.readthedocs.io/en/latest/advanced_tutorials/registry.html).

```python
import mmcv
import decord
import numpy as np
from mmcv.transforms import TRANSFORMS, BaseTransform, to_tensor
from mmaction.structures import ActionDataSample


@TRANSFORMS.register_module()
class VideoInit(BaseTransform):
    def transform(self, results):
        container = decord.VideoReader(results['filename'])
        results['total_frames'] = len(container)
        results['video_reader'] = container
        return results


@TRANSFORMS.register_module()
class VideoSample(BaseTransform):
    def __init__(self, clip_len, num_clips, test_mode=False):
        self.clip_len = clip_len
        self.num_clips = num_clips
        self.test_mode = test_mode

    def transform(self, results):
        total_frames = results['total_frames']
        interval = total_frames // self.clip_len

        if self.test_mode:
            # Make the sampling during testing deterministic
            np.random.seed(42)

        inds_of_all_clips = []
        for i in range(self.num_clips):
            bids = np.arange(self.clip_len) * interval
            offset = np.random.randint(interval, size=bids.shape)
            inds = bids + offset
            inds_of_all_clips.append(inds)

        results['frame_inds'] = np.concatenate(inds_of_all_clips)
        results['clip_len'] = self.clip_len
        results['num_clips'] = self.num_clips
        return results


@TRANSFORMS.register_module()
class VideoDecode(BaseTransform):
    def transform(self, results):
        frame_inds = results['frame_inds']
        container = results['video_reader']

        imgs = container.get_batch(frame_inds).asnumpy()
        imgs = list(imgs)

        results['video_reader'] = None
        del container

        results['imgs'] = imgs
        results['img_shape'] = imgs[0].shape[:2]
        return results


@TRANSFORMS.register_module()
class VideoResize(BaseTransform):
    def __init__(self, r_size):
        self.r_size = r_size

    def transform(self, results):
        imgs = [mmcv.imresize(img, (self.r_size[0], self.r_size[1]))
                for img in results['imgs']]
        results['imgs'] = imgs
        results['img_shape'] = imgs[0].shape[:2]
        return results


@TRANSFORMS.register_module()
class VideoCrop(BaseTransform):
    def __init__(self, c_size):
        self.c_size = c_size

    def transform(self, results):
        img_h, img_w = results['img_shape']
        center_x, center_y = img_w // 2, img_h // 2
        x1, x2 = center_x - self.c_size // 2, center_x + self.c_size // 2
        y1, y2 = center_y - self.c_size // 2, center_y + self.c_size // 2
        imgs = [img[y1:y2, x1:x2] for img in results['imgs']]
        results['imgs'] = imgs
        results['img_shape'] = imgs[0].shape[:2]
        return results


@TRANSFORMS.register_module()
class VideoFormat(BaseTransform):
    def transform(self, results):
        num_clips = results['num_clips']
        clip_len = results['clip_len']
        imgs = results['imgs']

        # [num_clips*clip_len, H, W, C]
        imgs = np.array(imgs)
        # [num_clips, clip_len, H, W, C]
        imgs = imgs.reshape((num_clips, clip_len) + imgs.shape[1:])
        # [num_clips, C, clip_len, H, W]
        imgs = imgs.transpose(0, 4, 1, 2, 3)

        results['imgs'] = imgs
        return results


@TRANSFORMS.register_module()
class VideoPack(BaseTransform):
    def __init__(self, meta_keys=('img_shape', 'num_clips', 'clip_len')):
        self.meta_keys = meta_keys

    def transform(self, results):
        packed_results = dict()
        inputs = to_tensor(results['imgs'])
        data_sample = ActionDataSample().set_gt_labels(results['label'])
        metainfo = {k: results[k] for k in self.meta_keys if k in results}
        data_sample.set_metainfo(metainfo)
        packed_results['inputs'] = inputs
        packed_results['data_samples'] = data_sample
        return packed_results
```

Below, we provide a code snippet (using `D32_1gwq35E.mp4 0` from the annotation file) to demonstrate how to use the pipeline.

```python
import os.path as osp
from mmengine.dataset import Compose

pipeline = [
    dict(type='VideoInit'),
    dict(type='VideoSample', clip_len=16, num_clips=1, test_mode=False),
    dict(type='VideoDecode'),
    dict(type='VideoResize', r_size=(256, 256)),
    dict(type='VideoCrop', c_size=224),
    dict(type='VideoFormat'),
    dict(type='VideoPack')
]

pipeline = Compose(transforms=pipeline)
data_prefix = 'data/kinetics400_tiny/train'
results = dict(filename=osp.join(data_prefix, 'D32_1gwq35E.mp4'), label=0)
packed_results = pipeline(results)

inputs = packed_results['inputs']
data_sample = packed_results['data_samples']

print('shape of the inputs: ', inputs.shape)

# Get metainfo of the inputs
print('image_shape: ', data_sample.img_shape)
print('num_clips: ', data_sample.num_clips)
print('clip_len: ', data_sample.clip_len)

# Get label of the inputs
print('label: ', data_sample.gt_labels.item)
```

The terminal output is as follows:

```shell
shape of the inputs:  torch.Size([1, 3, 16, 224, 224])
image_shape:  (224, 224)
num_clips:  1
clip_len:  16
label:  tensor([0])
```

## Step2: Build a Dataset and DataLoader

All `Dataset` classes in OpenMMLab must inherit from the `BaseDataset` class in `mmengine`. We can customize annotation loading process by overriding the `load_data_list` method. Additionally, we can add more information to the `results` dict that is passed as input to the `pipeline` by overriding the `get_data_info` method. For more detailed information about `BaseDataset` class, please refer to [MMEngine Tutorial](https://mmengine.readthedocs.io/en/latest/advanced_tutorials/basedataset.html).

```python
import os.path as osp
from mmengine.fileio import list_from_file
from mmengine.dataset import BaseDataset
from mmaction.registry import DATASETS


@DATASETS.register_module()
class DatasetZelda(BaseDataset):
    def __init__(self, ann_file, pipeline, data_root, data_prefix=dict(video=''),
                 test_mode=False, modality='RGB', **kwargs):
        self.modality = modality
        super(DatasetZelda, self).__init__(ann_file=ann_file, pipeline=pipeline, data_root=data_root,
                                           data_prefix=data_prefix, test_mode=test_mode,
                                           **kwargs)

    def load_data_list(self):
        data_list = []
        fin = list_from_file(self.ann_file)
        for line in fin:
            line_split = line.strip().split()
            filename, label = line_split
            label = int(label)
            filename = osp.join(self.data_prefix['video'], filename)
            data_list.append(dict(filename=filename, label=label))
        return data_list

    def get_data_info(self, idx: int) -> dict:
        data_info = super().get_data_info(idx)
        data_info['modality'] = self.modality
        return data_info
```

Next, we will demonstrate how to use dataset and dataloader to index data.

```python
from mmaction.registry import DATASETS

pipeline = [
    dict(type='VideoInit'),
    dict(type='VideoSample', clip_len=16, num_clips=1, test_mode=False),
    dict(type='VideoDecode'),
    dict(type='VideoResize', r_size=(256, 256)),
    dict(type='VideoCrop', c_size=224),
    dict(type='VideoFormat'),
    dict(type='VideoPack')
]

dataset = dict(type='DatasetZelda',
               ann_file='kinetics_tiny_train_video.txt',
               pipeline=pipeline,
               data_root='data/kinetics400_tiny/',
               data_prefix=dict(video='train'))
dataset = DATASETS.build(dataset)

packed_results = dataset[0]

inputs = packed_results['inputs']
data_sample = packed_results['data_samples']

print('shape of the inputs: ', inputs.shape)

# Get metainfo of the inputs
print('image_shape: ', data_sample.img_shape)
print('num_clips: ', data_sample.num_clips)
print('clip_len: ', data_sample.clip_len)

# Get label of the inputs
print('label: ', data_sample.gt_labels.item)

from mmengine.runner import Runner

BATCH_SIZE = 4

data_loader = dict(
    batch_size=BATCH_SIZE,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dataset)

data_loader = Runner.build_dataloader(dataloader=data_loader)

batched_packed_results = next(iter(data_loader))

batched_inputs = batched_packed_results['inputs']
batched_data_sample = batched_packed_results['data_samples']

assert len(batched_inputs) == BATCH_SIZE
assert len(batched_data_sample) == BATCH_SIZE
```

The terminal output should be the same as the one shown in the [Step1: Build a Pipeline](#step1-build-a-pipeline).

## Build a Recognizer




```python
import torch
import torch.nn as nn
import torchvision
from typing import Dict, Optional, Union
from mmengine.model import BaseModel, BaseModule, Sequential
from mmaction.registry import MODELS


@MODELS.register_module()
class RecognizerZelda(BaseModel):
    def __init__(self, backbone, cls_head, data_preprocessor):
        super(RecognizerZelda,
              self).__init__(data_preprocessor=data_preprocessor)

        self.backbone = MODELS.build(backbone)
        self.cls_head = MODELS.build(cls_head)
    
    def forward(self, inputs, data_samples=None, mode='tensor'):


@MODELS.register_module()
class BackBoneZelda(BaseModule):
    def __init__(self, init_cfg=None):
        if init_cfg is None:
            init_cfg = [dict(type='Kaiming', layer='Conv2d', mode='fan_out', nonlinearity="relu"),
                        dict(type='Constant', layer='BatchNorm2d', val=1, bias=0)]
        
        super(BackBoneZelda, self).__init__(init_cfg=init_cfg)

        self.conv1 = Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                                nn.BatchNorm2d(64), nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.conv = Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                               nn.BatchNorm2d(128), nn.ReLU())
    
    def forward(self, imgs):
        # imgs: [batch_size*num_views, 3, H, W]
        # features: [batch_size*num_views, 128, H//8, W//8]
        features = self.conv(self.maxpool(self.conv1(imgs))) 
        return features



```