# Customize Datasets

In this tutorial, we will introduce some methods about how to customize your own dataset by online conversion.

- [Customize Datasets](#customize-datasets)
  - [General understanding of the Dataset in MMAction2](#general-understanding-of-the-dataset-in-mmaction2)
  - [Customize new datasets](#customize-new-datasets)
  - [Customize keypoint format for PoseDataset](#customize-keypoint-format-for-posedataset)

## General understanding of the Dataset in MMAction2

MMAction2 provides specific Dataset class according to the task, e.g. `VideoDataset`/`RawframeDataset` for action recognition, `AVADataset` for spatio-temporal action detection, `PoseDataset` for skeleton-based action recognition. All these specific datasets only need to implement `get_data_info(self, idx)` to build a data list from the annotation file, while other functions are handled by the superclass. The following table shows the inherent relationship and the main function of the modules.

| Class Name                   | Functions                                                                                                                                                                    |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| MMAction2::VideoDataset      | `load_data_list(self)` <br> Build data list from the annotation file.                                                                                                        |
| MMAction2::BaseActionDataset | `get_data_info(self, idx)` <br> Given the `idx`, return the corresponding data sample from data list                                                                         |
| MMEngine::BaseDataset        | `__getitem__(self, idx)` <br> Given the `idx`, call `get_data_info` to get data sample, then call the `pipeline` to perform transforms and augmentation in `train_pipeline` or `val_pipeline` |

## Customize new datasets

For most scenarios, we don't need to customize a new dataset class, offline conversion is recommended way to use your data. But customizing a new dataset class is also easy in MMAction2. As above mentioned, a dataset for a specific task usually only needs to implement `load_data_list(self)` to generate the data list from the annotation file. It is worth noting that elements in the `data_list` are `dict` with fields required in the following pipeline.

Take `VideoDataset` as an example, `train_pipeline`/`val_pipeline` requires `'filename'` in `DecordInit` and `'label'` in `PackActionInput`, so data samples in the data list have 2 fields: `'filename'` and `'label'`.
you can refer to [customize pipeline](customize_pipeline.md) for more details about the pipeline.

```
data_list.append(dict(filename=filename, label=label))
```

While `AVADataset` is more complex, elements in the data list consist of several fields about video data, and it further overwrites `get_data_info(self, idx)` to convert keys, which are required in spatio-temporal action detection pipeline.

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

MMAction2 currently supports three kinds of keypoint formats: `coco`, `nturgb+d` and `openpose`. If your use one of them, just specify the corresponding format in the following modules:

For Graph Convolutional Networks, such as AAGCN, STGCN...

- transform: argument `dataset` in `JointToBone`.
- backbone: argument `graph_cfg` in Graph Convolutional Networks.

And for PoseC3D:

- transform: In `Flip`, specify `left_kp` and `right_kp` according to the keypoint symmetrical relationship, or remove the transform for asymmetric keypoints structure.
- transform: In `GeneratePoseTarget`, specify `skeletons`, `left_limb`, `right_limb` if `with_limb` is `true`, and `left_kp`, `right_kp` if `with_kp` is `true`.

For a custom format, you need to add a new graph layout into models and transforms, which defines the keypoints and their connection relationship.

Take the coco dataset as an example, we define a layout named `coco` in `Graph`, and set its `inward` as followed, which includes all connections between nodes, each connection is a pair of nodes from far to near. The order of connections does not matter. Other settings about coco are to set the number of nodes to 17, and set node 0 as the center node.

```python

self.num_node = 17
self.inward = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 5),
                (12, 6), (9, 7), (7, 5), (10, 8), (8, 6), (5, 0),
                (6, 0), (1, 0), (3, 1), (2, 0), (4, 2)]
self.center = 0
```

Similarly, we define the `pairs` in `JointToBone`, adding a bone of `(0, 0)` to align the number of bones to the nodes. The `pairs` of coco dataset is as followed, same as above mentioned, the order of pairs does not matter.

```python

self.pairs = ((0, 0), (1, 0), (2, 0), (3, 1), (4, 2), (5, 0),
              (6, 0), (7, 5), (8, 6), (9, 7), (10, 8), (11, 0),
              (12, 0), (13, 11), (14, 12), (15, 13), (16, 14))
```

For your custom format, just define the above setting as your graph structure, and specify in your config file as followed, we take `STGCN` as an example, assuming you already define a `custom_dataset` in `Graph` and `JointToBone`, and num_classes is n.

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
