_base_ = [
    '../../_base_/models/tsn_r50.py',
    '../../_base_/datasets/hmdb51_1x1x8_rgb.py',
    '../../_base_/schedules/sgd_50e.py', '../../_base_/default_runtime.py'
]

# model settings
model = dict(cls_head=dict(num_classes=51))

# optimizer
optimizer = dict(type='SGD', lr=0.025, momentum=0.9, weight_decay=0.0001)

# runtime settings
checkpoint_config = dict(interval=5)
work_dir = './work_dirs/tsn_r50_1x1x8_50e_hmdb51_kinetics400_rgb/'
load_from = 'https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_256p_1x1x8_100e_kinetics400_rgb/tsn_r50_256p_1x1x8_100e_kinetics400_rgb_20200817-883baf16.pth'  # noqa: E501
gpu_ids = range(0, 1)
