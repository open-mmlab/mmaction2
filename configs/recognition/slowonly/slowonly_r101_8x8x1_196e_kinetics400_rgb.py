_base_ = ['./slowonly_r50_8x8x1_256e_kinetics400_rgb.py']

# model settings
model = dict(backbone=dict(depth=101, pretrained=None))

# optimizer
optimizer = dict(
    type='SGD', lr=0.1, momentum=0.9,
    weight_decay=0.0001)  # this lr is used for 8 gpus
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_ratio=0.1,
    warmup_by_epoch=True,
    warmup_iters=34)
total_epochs = 196

# runtime settings
work_dir = './work_dirs/slowonly_r101_8x8x1_196e_kinetics400_rgb'
