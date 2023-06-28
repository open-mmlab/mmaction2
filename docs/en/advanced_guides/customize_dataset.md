# Customize Dataset

In this tutorial, we will introduce some methods about how to customize your own dataset by online conversion.

- [Customize Dataset](#customize-dataset)
  - [General understanding of the Dataset in MMAction2](#general-understanding-of-the-dataset-in-mmaction2)
  - [Customize new datasets](#customize-new-datasets)
  - [Customize keypoint format for PoseDataset](#customize-keypoint-format-for-posedataset)

## General understanding of the Dataset in MMAction2

MMAction2 provides task-specific `Dataset` class, e.g. `VideoDataset`/`RawframeDataset` for action recognition, `AVADataset` for spatio-temporal action detection, `PoseDataset` for skeleton-based action recognition. These task-specific datasets only require the implementation of `load_data_list(self)` for generating a data list from the annotation file. The remaining functions are automatically handled by the superclass (i.e., `BaseActionDataset` and `BaseDataset`). The following table shows the inheritance relationship and the main method of the modules.

| Class Name                     | Class Method                                                                                                                                                               |
| ------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `MMAction2::VideoDataset`      | `load_data_list(self)` <br> Build data list from the annotation file.                                                                                                      |
| `MMAction2::BaseActionDataset` | `get_data_info(self, idx)` <br> Given the `idx`, return the corresponding data sample from the data list.                                                                  |
| `MMEngine::BaseDataset`        | `__getitem__(self, idx)` <br> Given the `idx`, call `get_data_info` to get the data sample, then call the `pipeline` to perform transforms and augmentation in `train_pipeline` or `val_pipeline` . |

## Customize new datasets

Although offline conversion is the preferred method for utilizing your own data in most cases, MMAction2 offers a convenient process for creating a customized `Dataset` class. As mentioned previously, task-specific datasets only require the implementation of `load_data_list(self)` for generating a data list from the annotation file. It is noteworthy that the elements in the `data_list` are `dict` with fields that are essential for the subsequent processes in the `pipeline`.

Taking `VideoDataset` as an example, `train_pipeline`/`val_pipeline` require `'filename'` in `DecordInit` and `'label'` in `PackActionInputs`. Consequently, the data samples in the `data_list` must contain 2 fields: `'filename'` and `'label'`.
Please refer to [customize pipeline](customize_pipeline.md) for more details about the `pipeline`.

```
data_list.append(dict(filename=filename, label=label))
```

However, `AVADataset` is more complex, data samples in the `data_list` consist of several fields about the video data. Moreover, it overwrites `get_data_info(self, idx)` to convert keys that are indispensable in the spatio-temporal action detection pipeline.

```python

class AVADataset(BaseActionDataset):
  ...

  def load_data_list(self) -> List[dict]:
      ...
        video_info = dict(
            frame_dir=frame_dir,
            video_id=video_id,
            timestamp=int(timestamp),
            img_key=img_key,
            shot_info=shot_info,
            fps=self._FPS,
            ann=ann)
            data_list.append(video_info)
        data_list.append(video_info)
      return data_list

  def get_data_info(self, idx: int) -> dict:
      ...
      ann = data_info.pop('ann')
      data_info['gt_bboxes'] = ann['gt_bboxes']
      data_info['gt_labels'] = ann['gt_labels']
      data_info['entity_ids'] = ann['entity_ids']
      return data_info
```

## Customize keypoint format for PoseDataset

MMAction2 currently supports three keypoint formats: `coco`, `nturgb+d` and `openpose`. If you use one of these formats, you may simply specify the corresponding format in the following modules:

For Graph Convolutional Networks, such as AAGCN, STGCN, ...

- `pipeline`: argument `dataset` in `JointToBone`.
- `backbone`: argument `graph_cfg` in Graph Convolutional Networks.

For PoseC3D:

- `pipeline`: In `Flip`, specify `left_kp` and `right_kp` based on the symmetrical relationship between keypoints.
- `pipeline`: In `GeneratePoseTarget`, specify `skeletons`, `left_limb`, `right_limb` if `with_limb` is `True`, and `left_kp`, `right_kp` if `with_kp` is `True`.

If using a custom keypoint format, it is necessary to include a new graph layout in both the `backbone` and `pipeline`. This layout will define the keypoints and their connection relationship.

Taking the `coco` dataset as an example, we define a layout named `coco` in `Graph`. The `inward` connections of this layout comprise all node connections, with each **centripetal** connection consisting of a tuple of nodes. Additional settings for `coco` include specifying the number of nodes as `17` the `node 0` as the central node.

```python

self.num_node = 17
self.inward = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 5),
                (12, 6), (9, 7), (7, 5), (10, 8), (8, 6), (5, 0),
                (6, 0), (1, 0), (3, 1), (2, 0), (4, 2)]
self.center = 0
```

Similarly, we define the `pairs` in `JointToBone`, adding a bone of `(0, 0)` to align the number of bones to the nodes. The `pairs` of coco dataset are shown below, and the order of `pairs` in `JointToBone` is irrelevant.

```python

self.pairs = ((0, 0), (1, 0), (2, 0), (3, 1), (4, 2), (5, 0),
              (6, 0), (7, 5), (8, 6), (9, 7), (10, 8), (11, 0),
              (12, 0), (13, 11), (14, 12), (15, 13), (16, 14))
```

To use your custom keypoint format, simply define the aforementioned settings as your graph structure and specify them in your config file as shown below, In this example, we will use `STGCN`, with `n` denoting the number of classes and `custom_dataset` defined in `Graph` and `JointToBone`.

```python
model = dict(
  type='RecognizerGCN',
  backbone=dict(
      type='STGCN', graph_cfg=dict(layout='custom_dataset', mode='stgcn_spatial')),
  cls_head=dict(type='GCNHead', num_classes=n, in_channels=256))

train_pipeline = [
  ...
  dict(type='GenSkeFeat', dataset='custom_dataset'),
  ...]

val_pipeline = [
  ...
  dict(type='GenSkeFeat', dataset='custom_dataset'),
  ...]

test_pipeline = [
  ...
  dict(type='GenSkeFeat', dataset='custom_dataset'),
  ...]

```
