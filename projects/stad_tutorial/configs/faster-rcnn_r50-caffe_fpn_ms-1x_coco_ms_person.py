# Copyright (c) OpenMMLab. All rights reserved.
_base_ = './faster-rcnn_r50-caffe_fpn_ms-1x_coco.py'
model = dict(roi_head=dict(bbox_head=dict(num_classes=1)))

# take 2 epochs as an example
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=2, val_interval=1)

# learning rate
param_scheduler = [
    dict(type='ConstantLR', factor=1.0, by_epoch=False, begin=0, end=500)
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.0050, momentum=0.9, weight_decay=0.0001))

dataset_type = 'CocoDataset'
# modify metainfo
metainfo = {
    'classes': ('person', ),
    'palette': [
        (220, 20, 60),
    ]
}

# specify metainfo, dataset path
data_root = 'data/multisports/'

train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='annotations/multisports_det_anno_train.json',
        data_prefix=dict(img='rawframes/'),
        metainfo=metainfo))

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='annotations/multisports_det_anno_val.json',
        data_prefix=dict(img='rawframes/'),
        metainfo=metainfo))

test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='annotations/ms_infer_anno.json',
        data_prefix=dict(img='rawframes/'),
        metainfo=metainfo))

# specify annotaition file path, modify metric items
val_evaluator = dict(
    ann_file='data/multisports/annotations/multisports_det_anno_val.json',
    metric_items=['mAP_50', 'AR@100'],
    iou_thrs=[0.5],
)

test_evaluator = dict(
    ann_file='data/multisports/annotations/ms_infer_anno.json',
    metric_items=['mAP_50', 'AR@100'],
    iou_thrs=[0.5],
)

# specify pretrain checkpoint
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco-person/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth'  # noqa: E501
