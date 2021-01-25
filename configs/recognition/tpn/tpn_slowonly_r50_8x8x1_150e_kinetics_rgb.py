_base_ = ['./tpn_imagenet_pretrained_slowonly_r50_8x8x1_150e_kinetics_rgb.py']

# model settings
model = dict(backbone=dict(pretrained=None))

# runtime settings
work_dir = './work_dirs/tpn_slowonly_r50_8x8x1_150e_kinetics400_rgb'
