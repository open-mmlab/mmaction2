_base_ = [
    '../../_base_/models/audioonly_r50.py',
    '../../_base_/datasets/kinetics400_64x1x1_audio_feature.py',
    '../../_base_/default_runtime.py'
]

# optimizer
optimizer = dict(
    type='SGD', lr=2.0, momentum=0.9,
    weight_decay=0.0001)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0)
total_epochs = 100

# runtime settings
checkpoint_config = dict(interval=5)
log_config = dict(interval=1)
work_dir = ('./work_dirs/' +
            'audioonly_r50_64x1x1_100e_kinetics400_audio_feature/')
