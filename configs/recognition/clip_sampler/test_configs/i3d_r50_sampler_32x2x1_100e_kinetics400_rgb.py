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
        audio_recognizer_config=dict(
            type='AudioRecognizer',
            backbone=dict(
                type='ResNet', depth=18, in_channels=1, norm_eval=False),
            cls_head=dict(
                type='AudioTSNHead',
                num_classes=400,
                in_channels=512,
                dropout_ratio=0.5,
                init_std=0.01)),
        pretrained_audio='tsn_r18_64x1x1_100e_kinetics400_audio_feature.pth',
        mv_recognizer_config=None,
        pretrained_mv=None,
        if_recognizer_config=None,
        pretrained_if=None,
    ))
# model training and testing settings
train_cfg = dict(aux_info=['mvs', 'i_frames', 'audios'])
test_cfg = dict(average_clips='prob')
# dataset settings
dataset_type = 'AudioVisualDataset'
data_root_val = 'data/kinetics400/videos_val'
audio_prefix = 'data/kinetics400/audios_feature'
ann_file_test = 'data/kinetics400/clean_kinetics400_val_list_audio_feature.txt'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
test_pipeline = [
    dict(type='PyAVInit'),
    dict(type='LoadAudioFeature'),
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=30),
    dict(type='PyAVDecodeSideData'),
    dict(type='AudioFeatureSelector'),
    dict(type='Resize', scale=(-1, 256)),
    # dict(type='Resize', scale=(16, 16), field='mvs'),
    dict(type='ThreeCrop', crop_size=256),
    # dict(type='CenterCrop', crop_size=256, field='i_frames'),
    dict(type='Normalize', **img_norm_cfg),
    # dict(type='Normalize', **img_norm_cfg, field='i_frames'),
    dict(type='FormatShape', input_format='NCTHW'),
    # dict(type='FormatShape', input_format='NCHW', field='mvs'),
    # dict(type='FormatShape', input_format='NCHW', field='i_frames'),
    dict(type='FormatAudioShape', input_format='NCTF'),
    dict(
        type='Collect',
        # keys=['imgs', 'audios', 'mvs', 'i_frames', 'label'],
        keys=['imgs', 'audios', 'label'],
        meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'audios', 'label'])
]

data = dict(
    videos_per_gpu=1,
    workers_per_gpu=0,
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        video_prefix=data_root_val,
        audio_prefix=audio_prefix,
        start_index=1,
        pipeline=test_pipeline))
# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
