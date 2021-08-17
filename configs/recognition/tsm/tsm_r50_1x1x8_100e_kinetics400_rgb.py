_base_ = ['./tsm_r50_1x1x8_50e_kinetics400_rgb.py']

optimizer_config = dict(grad_clip=dict(max_norm=20, norm_type=2))
lr_config = dict(policy='step', step=[40, 80])
total_epochs = 100
work_dir = './work_dirs/tsm_r50_1x1x8_100e_kinetics400_rgb/'
