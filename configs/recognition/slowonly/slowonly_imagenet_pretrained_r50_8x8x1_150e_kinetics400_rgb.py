_base_ = [
    '../../_base_/models/slowonly_r50.py',
    '../../_base_/datasets/kinetics400_8x8x1_rgb.py',
    '../../_base_/default_runtime.py'
]

# optimizer
optimizer = dict(
    type='SGD', lr=0.01, momentum=0.9,
    weight_decay=0.0001)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    step=[90, 130],
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=10)
total_epochs = 150

# runtime settings
checkpoint_config = dict(interval=4)
work_dir = ('./work_dirs/slowonly_imagenet_pretrained_r50_8x8x1_150e'
            '_kinetics400_rgb')
find_unused_parameters = False
