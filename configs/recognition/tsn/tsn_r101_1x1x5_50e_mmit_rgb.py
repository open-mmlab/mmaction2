_base_ = [
    '../../_base_/datasets/mmit_1x1x5_rgb.py',
    '../../_base_/schedules/sgd_tsm_50e.py', '../../_base_/default_runtime.py'
]

# model settings
model = dict(
    type='Recognizer2D',
    backbone=dict(
        type='ResNet',
        pretrained='torchvision://resnet101',
        depth=101,
        norm_eval=False),
    cls_head=dict(
        type='TSNHead',
        num_classes=313,
        in_channels=2048,
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        loss_cls=dict(type='BCELossWithLogits', loss_weight=160.0),
        dropout_ratio=0.5,
        init_std=0.01,
        multi_class=True,
        label_smooth_eps=0))
# model training and testing settings
train_cfg = None
test_cfg = dict(average_clips=None)

# runtime settings
checkpoint_config = dict(interval=5)
work_dir = './work_dirs/tsn_r101_1x1x5_50e_mmit_rgb/'
