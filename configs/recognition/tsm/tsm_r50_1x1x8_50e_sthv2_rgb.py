_base_ = [
    '../../_base_/models/tsm_r50.py',
    '../../_base_/datasets/sthv2_1x1x8_rgb.py',
    '../../_base_/schedules/sgd_tsm_50e.py', '../../_base_/default_runtime.py'
]

# model settings
model = dict(cls_head=dict(num_classes=174))

# dataset settings
data = dict(videos_per_gpu=6, workers_per_gpu=4)

# optimizer
optimizer = dict(
    lr=0.0075,  # this lr is used for 8 gpus
    weight_decay=0.0005)

# runtime settings
work_dir = './work_dirs/tsm_r50_1x1x8_50e_sthv2_rgb/'
