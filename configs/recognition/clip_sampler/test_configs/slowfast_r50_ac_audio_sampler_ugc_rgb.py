# model settings
model = dict(
    type='Recognizer3DSampler',
    backbone=dict(
        type='ResNet3dSlowFast',
        pretrained=None,
        resample_rate=4,  # tau
        speed_ratio=4,  # alpha
        channel_ratio=8,  # beta_inv
        slow_pathway=dict(
            type='resnet3d',
            depth=50,
            pretrained=None,
            lateral=True,
            fusion_kernel=7,
            conv1_kernel=(1, 7, 7),
            dilations=(1, 1, 1, 1),
            conv1_stride_t=1,
            pool1_stride_t=1,
            inflate=(0, 0, 1, 1),
            norm_eval=False),
        fast_pathway=dict(
            type='resnet3d',
            depth=50,
            pretrained=None,
            lateral=False,
            base_channels=8,
            conv1_kernel=(5, 7, 7),
            conv1_stride_t=1,
            pool1_stride_t=1,
            norm_eval=False)),
    cls_head=dict(
        type='SlowFastHead',
        in_channels=2304,  # 2048+256
        num_classes=212,
        spatial_type='avg',
        dropout_ratio=0.5),
    sampler=dict(
        type='ACSampler',
        top_k=10,
        audio_recognizer_config=dict(
            type='AudioRecognizer',
            backbone=dict(
                type='ResNet', depth=18, in_channels=1, norm_eval=False),
            cls_head=dict(
                type='AudioTSNHead',
                num_classes=212,
                in_channels=512,
                dropout_ratio=0.5,
                init_std=0.01)),
        pretrained_audio='epoch_200.pth',
        mv_recognizer_config=None,
        pretrained_mv=None,
        if_recognizer_config=None,
        pretrained_if=None,
    ))
# model training and testing settings
# train_cfg = dict(aux_info=['mvs', 'i_frames', 'audios'])
train_cfg = dict(aux_info=['audios'])
test_cfg = dict(average_clips='prob')
# dataset settings
dataset_type = 'AudioVisualDataset'
data_root_val = 'data/ugc/rawframes'
audio_prefix = 'data/ugc/audio_feature'
ann_file_test = 'data/ugc/ugc_val_list_rawframes.txt'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
test_pipeline = [
    # dict(type='PyAVInit'),
    dict(type='LoadAudioFeature'),
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=30),
    # dict(type='PyAVDecodeSideData'),
    dict(type='RawFrameDecode'),
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
    workers_per_gpu=4,
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        # video_prefix=data_root_val,
        data_prefix=data_root_val,
        audio_prefix=audio_prefix,
        start_index=1,
        pipeline=test_pipeline))
# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
