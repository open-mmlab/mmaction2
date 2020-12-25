_base_ = [
    '../../_base_/models/slowonly_r50.py',
    '../../_base_/datasets/gym99_4x16x1_rgb.py',
    '../../_base_/default_runtime.py'
]

# optimizer
optimizer = dict(
    type='SGD', lr=0.03, momentum=0.9,
    weight_decay=0.0001)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[90, 110])
total_epochs = 120

# runtime settings
work_dir = './work_dirs/slowonly_imagenet_pretrained_r50_4x16x1_120e_gym99_rgb'
find_unused_parameters = False
