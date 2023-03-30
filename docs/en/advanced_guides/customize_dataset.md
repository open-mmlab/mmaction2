# Customize Datasets

In this tutorial, we will introduce some methods about how to customize your own dataset by online conversion

## Custom new dataset

You can write a new Dataset class inherited from [BaseDataset](/mmaction/datasets/base.py), and overwrite three methods
`load_annotations(self)`, `evaluate(self, results, metrics, logger)` and `dump_results(self, results, out)`,
like [RawframeDataset](/mmaction/datasets/rawframe_dataset.py), [VideoDataset](/mmaction/datasets/video_dataset.py) or [ActivityNetDataset](/mmaction/datasets/activitynet_dataset.py).

## Custom keypoint format for PoseDataset

MMAction2 currently supports three kinds of keypoint formats: `coco`, `nturgb+d` and `openpose`. If your use one of them, just specify the corresponding format in the following modules:

For Graph Convolutional Networks, such as AAGCN, STGCN...

- transform: argument `dataset` in `JointToBone`.
- backbone: argument `graph_cfg` in Graph Convolutional Networks.

And for PoseC3D:

- transform: In `Flip`, specify `left_kp` and `right_kp` according to the keypoint symmetrical relationship, or remove the transform for asymmetric keypoints structure.
- transform: In `GeneratePoseTarget`, specify `skeletons`, `left_limb`, `right_limb` if `with_limb` is `true`, and `left_kp`, `right_kp` if `with_kp` is `true`.

For a custom format, you need to add a new graph layout into models and transforms, which defines the keypoints and their connection relationship.

Take the coco dataset as an example, we define a layout named `coco` in `Graph`, and set its `inward` as followed, which includes all connections between nodes, each connection is a pair of nodes from far to near. The order of connections does not matter. Other settings about coco are to set the number of nodes to 17, and set node 0 as the center node.

````
```
self.num_node = 17
self.inward = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 5),
                (12, 6), (9, 7), (7, 5), (10, 8), (8, 6), (5, 0),
                (6, 0), (1, 0), (3, 1), (2, 0), (4, 2)]
self.center = 0
```
````

Similarly, we define the `pairs` in `JointToBone`, adding a bone of `(0, 0)` to align the number of bones to the nodes. The `pairs` of coco dataset is as followed, same as above mentioned, the order of pairs does not matter.

````
```
self.pairs = ((0, 0), (1, 0), (2, 0), (3, 1), (4, 2), (5, 0),
              (6, 0), (7, 5), (8, 6), (9, 7), (10, 8), (11, 0),
              (12, 0), (13, 11), (14, 12), (15, 13), (16, 14))
```
````

For your custom format, just define the above setting as your graph structure, and specify in your config file as followed, we take `STGCN` as an example, assuming you already define a `custom_dataset` in `Graph` and `JointToBone`, and num_classes is n.

````
```
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
````
