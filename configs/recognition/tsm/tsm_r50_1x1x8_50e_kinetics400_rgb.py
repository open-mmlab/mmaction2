_base_ = [
    '../../_base_/models/tsm_r50.py',
    '../../_base_/datasets/kinetics400_1x1x8_rgb.py',
    '../../_base_/schedules/sgd_tsm_50e.py', '../../_base_/default_runtime.py'
]

# runtime settings
checkpoint_config = dict(interval=5)
work_dir = './work_dirs/tsm_r50_1x1x8_100e_kinetics400_rgb/'
