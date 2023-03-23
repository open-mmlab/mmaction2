# Tutorial 3: Adding New Dataset

In this tutorial, we will introduce some methods about how to customize your own dataset by reorganizing data and mixing dataset for the project.

<!-- TOC -->

- [Tutorial 3: Adding New Dataset](#tutorial-3-adding-new-dataset)
  - [Customize Datasets by Reorganizing Data](#customize-datasets-by-reorganizing-data)
    - [Reorganize datasets to existing format](#reorganize-datasets-to-existing-format)
    - [An example of a custom dataset](#an-example-of-a-custom-dataset)
  - [Customize Dataset by Mixing Dataset](#customize-dataset-by-mixing-dataset)
    - [Repeat dataset](#repeat-dataset)

<!-- TOC -->

## Customize Datasets by Reorganizing Data

### Reorganize datasets to existing format

The simplest way is to convert your dataset to existing dataset formats.

You can refer to the annotation format according to your task

- Action Recognition

  The task support two kinds of annotation format

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

- Action localization

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

- Skeleton based action recognition

  The task recognizes the action class based on the skeleton sequence (time sequence of keypoints). We provide tutorial on preparing datasets for different conditions.

  Assuming that you have a RGB video dataset, there are four steps for you to train a skeleton based action recognition model on it.

  ```
    Step 1: Extract skeleton data from RGB video

    Step 2: Format skeleton data to MMAction2's specification

    Step 3: Specify your custom dataset in the config file

    Step 4: Train model based on custom config
  ```

  - Option 1: Build from RGB video dataset

    MMAction2 provides a script to perform step 1 and step 2, the script extracts keypoint data based on MMDetection and MMPose. You can refer to the [doc](/configs/skeleton/posec3d/custom_dataset_training.md) for detailed instructions.

  - Option 2: Build from the existing keypoint dataset

    Assuming that you have extracted keypoint data for your video dataset, you need to convert your keypoint data to MMACtion2's specification, as followed:

    ```
    {
        "split":
            {
                'xsub_train':
                    ['S001C001P001R001A001', ...],
                'xsub_val':
                    ['S001C001P003R001A001', ...],
                ...
            }

        "annotations:
            [
                {
                    {
                        'frame_dir': 'S001C001P001R001A001',
                        'label': 0,
                        'img_shape': (1080, 1920),
                        'original_shape': (1080, 1920),
                        'total_frames': 103,   # num_frames
                        'keypoint': array([[[[1032. ,  334.8], ...]]])  # array with shape (num_person, num_frames, num_keypoints, 2)
                        'keypoint_score': array([[[0.934 , 0.9766, ...]]])  # array with shape (num_person, num_frames, num_keypoints)
                    },
                    {
                        'frame_dir': 'S001C001P003R001A001',
                        ...
                    },
                    ...

                }
            ]
    }
    ```

  Assume the annotation file at `data/posec3d/ntu60_2d.pkl`, then you can use the dataset with below configurations:

  ```
  train_dataloader = dict(
      ...
      dataset=dict(,
          dataset=dict(
              type=PoseDataset,
              ann_file='data/posec3d/ntu60_2d.pkl',
              split='xsub_train',
              pipeline=train_pipeline)))

  val_dataloader = dict(
      ...
      dataset=dict(
          type=PoseDataset,
          ann_file='data/posec3d/ntu60_2d.pkl,
          split='xsub_val',
          pipeline=val_pipeline,
          test_mode=True))

  test_dataloader = val_dataloader
  ```

  Split annotation file to train and validation part is also supported, you only need to specify the respective file path to `ann_file` and ignore the `split` argument in dataset.

  In more complicated cases, your skeleton data may have different keypoint number from our default settings (17 points for coco format), which requires further customization.

  Currently, we support three mainstream keypoint formats: `coco` format for default, `nturgb+d` for ntu original annotation, and `openpose`. If your use one of them, just specify the corresponding dataset in the following modules:

  For Graph Convolutional Networks, such as AAGCN, STGCN ...

  - transform: argument `dataset` in `JointToBone`.

  - backbone: argument `graph_cfg` in Graph Convolutional Networks.

  For PoseC3D:

  - transform: specify `left_kp` and `right_kp` in `Flip` according to the keypoint symmetrical relationship, or remove the transform for asymmetric keypoints structure.

  - transform: specify `skeletons`, `left_limb`, `right_limb` if `with_limb` is `true`, and `left_kp`, `right_kp` if `with_kp` is `true`.

  For the skeleton format not included, you could add a new graph layout to the above modules, which defines the keypoint and their connection relationship.

  Remaining steps about modify `num_classes` are the same as other tasks.

- Spatio-temporal action detection

  ```
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
    workers_per_gpu=2,
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
