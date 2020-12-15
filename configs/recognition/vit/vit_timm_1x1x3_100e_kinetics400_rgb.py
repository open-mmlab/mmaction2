# model settings
model = dict(
    type='Recognizer2D',
    backbone=dict(type='ViT_Timm'),
    cls_head=dict(type='TSN_VIT_Head', num_classes=400, in_channels=1000))
# model training and testing settings
train_cfg = None
test_cfg = dict(average_clips='score')
# dataset settings

mc_cfg = dict(
    server_list_cfg='/mnt/lustre/share/memcached_client/server_list.conf',
    client_cfg='/mnt/lustre/share/memcached_client/client.conf',
    sys_path='/mnt/lustre/share/pymc/py3')

dataset_type = 'RawframeDataset'
data_root = 'data/kinetics400/rawframes_train'
data_root_val = 'data/kinetics400/rawframes_val'
ann_file_train = 'data/kinetics400/kinetics400_train_list_rawframes.txt'
ann_file_val = 'data/kinetics400/kinetics400_val_list_rawframes.txt'
ann_file_test = 'data/kinetics400/kinetics400_val_list_rawframes.txt'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    # dict(type='PyAVInit', io_backend='memcached', **mc_cfg),
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=3),
    # dict(type='PyAVDecode'),
    dict(
        type='RawFrameDecode',
        decoding_backend='turbojpeg',
        io_backend='memcached',
        **mc_cfg),
    dict(type='Resize', scale=(-1, 224)),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.875, 0.75, 0.66),
        random_crop=False,
        max_wh_scale_gap=1),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    # dict(type='PyAVInit', io_backend='memcached', **mc_cfg),
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=3,
        test_mode=True),
    # dict(type='PyAVDecode'),
    dict(
        type='RawFrameDecode',
        decoding_backend='turbojpeg',
        io_backend='memcached',
        **mc_cfg),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    # dict(type='PyAVInit', io_backend='memcached', **mc_cfg),
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=25,
        test_mode=True),
    # dict(type='PyAVDecode'),
    dict(
        type='RawFrameDecode',
        decoding_backend='turbojpeg',
        io_backend='memcached',
        **mc_cfg),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=256),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=32,  # 32
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline))
# optimizer
optimizer = dict(
    type='SGD',
    lr=0.006,
    momentum=0.9,
    # 4096 / (32b * 8g * 3c) = 5.3; 5.3 * 6e-4 = 3e-3
    weight_decay=0.1)  # this lr is used for 16 gpus;
# very large weight_decay
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_ratio=0.001,
    warmup_by_epoch=True,
    warmup_iters=20)

total_epochs = 200
checkpoint_config = dict(interval=5)
evaluation = dict(
    interval=5, metrics=['mean_class_accuracy', 'top_k_accuracy'])
log_config = dict(
    interval=20,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/vit_timm_1x1x3_100e_kinetics400_rgb/'
load_from = None
resume_from = None
workflow = [('train', 1)]
find_unused_parameters = True
