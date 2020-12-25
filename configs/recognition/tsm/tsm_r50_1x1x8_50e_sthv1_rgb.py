_base_ = [
    '../../_base_/models/tsm_r50.py',
    '../../_base_/datasets/sthv1_1x1x8_rgb.py',
    '../../_base_/schedules/sgd_tsm_50e.py', '../../_base_/default_runtime.py'
]

# model settings
model = dict(cls_head=dict(num_classes=174))

# optimizer
optimizer = dict(weight_decay=0.0005)

# runtime settings
work_dir = './work_dirs/tsm_r50_1x1x8_50e_sthv1_rgb/'
