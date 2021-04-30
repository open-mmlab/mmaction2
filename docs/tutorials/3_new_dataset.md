# Tutorial 3: Adding New Dataset

In this tutorial, we will introduce some methods about how to customize your own dataset by reorganizing data and mixing dataset for the project.

<!-- TOC -->

- [Customize Datasets by Reorganizing Data](#customize-datasets-by-reorganizing-data)
  - [Reorganize datasets to existing format](#reorganize-datasets-to-existing-format)
  - [An example of a custom dataset](#an-example-of-a-custom-dataset)
- [Customize Dataset by Mixing Dataset](#customize-dataset-by-mixing-dataset)
  - [Repeat dataset](#repeat-dataset)

<!-- TOC -->

## Customize Datasets by Reorganizing Data

### Reorganize datasets to existing format

The simplest way is to convert your dataset to existing dataset formats (RawframeDataset or VideoDataset).

There are three kinds of annotation files.

- rawframe annotation

  The annotation of a rawframe dataset is a text file with multiple lines,
  and each line indicates `frame_directory` (relative path) of a video,
  `total_frames` of a video and the `label` of a video, which are split by a whitespace.

  Here is an example.

  ```
  some/directory-1 163 1
  some/directory-2 122 1
  some/directory-3 258 2
  some/directory-4 234 2
  some/directory-5 295 3
  some/directory-6 121 3
  ```

- video annotation

  The annotation of a video dataset is a text file with multiple lines,
  and each line indicates a sample video with the `filepath` (relative path) and `label`,
  which are split by a whitespace.

  Here is an example.

  ```
  some/path/000.mp4 1
  some/path/001.mp4 1
  some/path/002.mp4 2
  some/path/003.mp4 2
  some/path/004.mp4 3
  some/path/005.mp4 3
  ```

- ActivityNet annotation

  The annotation of ActivityNet dataset is a json file. Each key is a video name
  and the corresponding value is the meta data and annotation for the video.

  Here is an example.

  ```
  {
    "video1": {
        "duration_second": 211.53,
        "duration_frame": 6337,
        "annotations": [
            {
                "segment": [
                    30.025882995319815,
                    205.2318595943838
                ],
                "label": "Rock climbing"
            }
        ],
        "feature_frame": 6336,
        "fps": 30.0,
        "rfps": 29.9579255898
    },
    "video2": {
        "duration_second": 26.75,
        "duration_frame": 647,
        "annotations": [
            {
                "segment": [
                    2.578755070202808,
                    24.914101404056165
                ],
                "label": "Drinking beer"
            }
        ],
        "feature_frame": 624,
        "fps": 24.0,
        "rfps": 24.1869158879
    }
  }
  ```

There are two ways to work with custom datasets.

- online conversion

  You can write a new Dataset class inherited from [BaseDataset](/mmaction/datasets/base.py), and overwrite three methods
  `load_annotations(self)`, `evaluate(self, results, metrics, logger)` and `dump_results(self, results, out)`,
  like [RawframeDataset](/mmaction/datasets/rawframe_dataset.py), [VideoDataset](/mmaction/datasets/video_dataset.py) or [ActivityNetDataset](/mmaction/datasets/activitynet_dataset.py).

- offline conversion

  You can convert the annotation format to the expected format above and save it to
  a pickle or json file, then you can simply use `RawframeDataset`, `VideoDataset` or `ActivityNetDataset`.

After the data pre-processing, the users need to further modify the config files to use the dataset.
Here is an example of using a custom dataset in rawframe format.

In `configs/task/method/my_custom_config.py`:

```python
...
# dataset settings
dataset_type = 'RawframeDataset'
data_root = 'path/to/your/root'
data_root_val = 'path/to/your/root_val'
ann_file_train = 'data/custom/custom_train_list.txt'
ann_file_val = 'data/custom/custom_val_list.txt'
ann_file_test = 'data/custom/custom_val_list.txt'
...
data = dict(
    videos_per_gpu=32,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        ...),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        ...),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        ...))
...
```

We use this way to support Rawframe dataset.

### An example of a custom dataset

Assume the annotation is in a new format in text files, and the image file name is of template like `img_00005.jpg`
The video annotations are stored in text file `annotation.txt` as following

```
directory,total frames,class
D32_1gwq35E,299,66
-G-5CJ0JkKY,249,254
T4h1bvOd9DA,299,33
4uZ27ivBl00,299,341
0LfESFkfBSw,249,186
-YIsNpBEx6c,299,169
```

We can create a new dataset in `mmaction/datasets/my_dataset.py` to load the data.

```python
import copy
import os.path as osp

import mmcv

from .base import BaseDataset
from .builder import DATASETS


@DATASETS.register_module()
class MyDataset(BaseDataset):

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_prefix=None,
                 test_mode=False,
                 filename_tmpl='img_{:05}.jpg'):
        super(MyDataset, self).__init__(ann_file, pipeline, test_mode)

        self.filename_tmpl = filename_tmpl

    def load_annotations(self):
        video_infos = []
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                if line.startswith("directory"):
                    continue
                frame_dir, total_frames, label = line.split(',')
                if self.data_prefix is not None:
                    frame_dir = osp.join(self.data_prefix, frame_dir)
                video_infos.append(
                    dict(
                        frame_dir=frame_dir,
                        total_frames=int(total_frames),
                        label=int(label)))
        return video_infos

    def prepare_train_frames(self, idx):
        results = copy.deepcopy(self.video_infos[idx])
        results['filename_tmpl'] = self.filename_tmpl
        return self.pipeline(results)

    def prepare_test_frames(self, idx):
        results = copy.deepcopy(self.video_infos[idx])
        results['filename_tmpl'] = self.filename_tmpl
        return self.pipeline(results)

    def evaluate(self,
                 results,
                 metrics='top_k_accuracy',
                 topk=(1, 5),
                 logger=None):
        pass
```

Then in the config, to use `MyDataset` you can modify the config as the following

```python
dataset_A_train = dict(
    type='MyDataset',
    ann_file=ann_file_train,
    pipeline=train_pipeline
)
```

## Customize Dataset by Mixing Dataset

MMAction2 also supports to mix dataset for training. Currently it supports to repeat dataset.

### Repeat dataset

We use `RepeatDataset` as wrapper to repeat the dataset. For example, suppose the original dataset as `Dataset_A`,
to repeat it, the config looks like the following

```python
dataset_A_train = dict(
        type='RepeatDataset',
        times=N,
        dataset=dict(  # This is the original config of Dataset_A
            type='Dataset_A',
            ...
            pipeline=train_pipeline
        )
    )
```
