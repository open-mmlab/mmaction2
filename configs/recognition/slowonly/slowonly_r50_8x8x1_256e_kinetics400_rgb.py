_base_ = [
    '../../_base_/models/slowonly_r50.py',
    '../../_base_/datasets/kinetics400_8x8x1_rgb.py',
    '../../_base_/default_runtime.py'
]

# model settings
model = dict(backbone=dict(pretrained=None))

# optimizer
optimizer = dict(
    type='SGD', lr=0.1, momentum=0.9,
    weight_decay=0.0001)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0)
total_epochs = 256

# runtime settings
checkpoint_config = dict(interval=4)
work_dir = './work_dirs/slowonly_r50_8x8x1_256e_kinetics400_rgb'
find_unused_parameters = False
