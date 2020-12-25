_base_ = [
    '../../_base_/models/slowonly_r50.py',
    '../../_base_/datasets/kinetics400_4x16x1_flow.py',
    '../../_base_/default_runtime.py'
]

# model settings
model = dict(backbone=dict(in_channels=2, with_pool2=False))

# optimizer
optimizer = dict(
    type='SGD', lr=0.06, momentum=0.9,
    weight_decay=0.0001)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=34)
total_epochs = 256

# runtime settings
checkpoint_config = dict(interval=4)
work_dir = './work_dirs/slowonly_r50_4x16x1_256e_kinetics400_flow'
find_unused_parameters = False
