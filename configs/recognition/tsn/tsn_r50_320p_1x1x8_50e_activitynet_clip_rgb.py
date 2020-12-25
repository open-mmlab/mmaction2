_base_ = [
    '../../_base_/models/tsn_r50.py',
    '../../_base_/datasets/activitynet_1x1x8_rgb.py',
    '../../_base_/schedules/sgd_50e.py', '../../_base_/default_runtime.py'
]
# model settings
model = dict(
    backbone=dict(
        pretrained='modelzoo/tsn_r50_320p_1x1x8_100e_kinetics400_rgb.pth'),
    cls_head=dict(num_classes=200, dropout_ratio=0.8))

# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)

# runtime settings
work_dir = './work_dirs/tsn_r50_320p_1x1x8_50e_activitynet_clip_rgb/'
workflow = [('train', 5)]
