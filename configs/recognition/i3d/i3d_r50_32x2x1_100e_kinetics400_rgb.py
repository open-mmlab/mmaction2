_base_ = [
    '../../_base_/models/i3d_r50.py',
    '../../_base_/datasets/kinetics400_32x2x1_rgb.py',
    '../../_base_/schedules/sgd_100e.py', '../../_base_/default_runtime.py'
]

# runtime settings
checkpoint_config = dict(interval=5)
work_dir = './work_dirs/i3d_r50_32x2x1_100e_kinetics400_rgb/'
