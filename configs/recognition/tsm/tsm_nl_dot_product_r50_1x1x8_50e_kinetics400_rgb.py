_base_ = [
    '../../_base_/models/tsm_r50.py',
    '../../_base_/datasets/kinetics400_1x1x8_rgb.py',
    '../../_base_/schedules/sgd_tsm_50e.py', '../../_base_/default_runtime.py'
]

# model settings
model = dict(
    backbone=dict(
        non_local=((0, 0, 0), (1, 0, 1, 0), (1, 0, 1, 0, 1, 0), (0, 0, 0)),
        non_local_cfg=dict(
            sub_sample=True,
            use_scale=False,
            norm_cfg=dict(type='BN3d', requires_grad=True),
            mode='dot_product')))

# runtime settings
work_dir = './work_dirs/tsm_nl_gaussian_r50_1x1x8_50e_kinetics400_rgb/'
