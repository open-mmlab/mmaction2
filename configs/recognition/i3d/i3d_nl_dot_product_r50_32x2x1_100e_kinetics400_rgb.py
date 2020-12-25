_base_ = [
    '../../_base_/models/i3d_r50.py',
    '../../_base_/datasets/kinetics400_32x2x1_rgb.py',
    '../../_base_/schedules/sgd_100e.py', '../../_base_/default_runtime.py'
]

# model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(
        non_local=((0, 0, 0), (0, 1, 0, 1), (0, 1, 0, 1, 0, 1), (0, 0, 0)),
        non_local_cfg=dict(
            sub_sample=True,
            use_scale=False,
            norm_cfg=dict(type='BN3d', requires_grad=True),
            mode='dot_product')))

# runtime settings
checkpoint_config = dict(interval=5)
work_dir = './work_dirs/i3d_nl_dot_product_r50_32x2x1_100e_kinetics400_rgb/'
