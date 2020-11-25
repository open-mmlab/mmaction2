# model settings
model = dict(
    type='Recognizer3DSampler',
    backbone=dict(
        type='ResNet3d',
        pretrained2d=True,
        pretrained='torchvision://resnet50',
        depth=50,
        conv_cfg=dict(type='Conv3d'),
        norm_eval=False,
        inflate=((1, 1, 1), (1, 0, 1, 0), (1, 0, 1, 0, 1, 0), (0, 1, 0)),
        zero_init_residual=False),
    cls_head=dict(
        type='I3DHead',
        num_classes=400,
        in_channels=2048,
        spatial_type='avg',
        dropout_ratio=0.5,
        init_std=0.01),
    sampler=dict(
        type='ACSampler',
        top_k=10,
        pretrained_audio='./audio_ac_sampler.pth',
        pretrained_mv='./mv_ac_sampler.pth',
        pretrained_if='./if_ac_sampler.pth',
    ))
# model training and testing settings
train_cfg = None
test_cfg = dict(average_clips='prob')
# dataset settings
dataset_type = 'AudioVisualDataset'
data_root = 'data/kinetics400/rawframes_train'
data_root_val = 'data/kinetics400/rawframes_val'
audio_prefix = 'data/kinetics400/audio_features'
ann_file_train = 'data/kinetics400/kinetics400_train_list_rawframes.txt'
ann_file_val = 'data/kinetics400/kinetics400_val_list_rawframes.txt'
ann_file_test = 'data/kinetics400/kinetics400_val_list_rawframes.txt'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
test_pipeline = [
    dict(type='PyAVInit'),
    dict(type='LoadAudioFeature'),
    dict(
        type='DenseSampleFrames', clip_len=32, frame_interval=2, num_clips=30),
    dict(type='PyAVDecode'),
    dict(type='PyAVMotionVectorDecode'),
    dict(type='PyAVIFrameDecode'),
    dict(type='AudioFeatureSelector'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='FormatAudioShape', input_format='NCTF'),
    dict(
        type='Collect',
        keys=['imgs', 'audios', 'mvs', 'i_frames', 'label'],
        meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'audios', 'mvs', 'i_frames', 'label'])
]

data = dict(
    videos_per_gpu=8,
    workers_per_gpu=4,
    test=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=test_pipeline))
# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = ''
resume_from = None
