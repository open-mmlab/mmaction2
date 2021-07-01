_base_ = [
    '../../_base_/models/slowfast_r50.py', '../../_base_/default_runtime.py'
]

dataset_type = 'RawframeDataset'
# data_root = 'data/kinetics400/rawframes_train'
# data_root_val = 'data/kinetics400/rawframes_val'
# ann_file_train = 'data/kinetics400/kinetics400_train_list_rawframes.txt'
# ann_file_val = 'data/kinetics400/kinetics400_val_list_rawframes.txt'
# ann_file_test = 'data/kinetics400/kinetics400_val_list_rawframes.txt'
data_root = '/mnt/lustre21/DATAshare2/duanhaodong/Kinetics400/kinetics_400_train_SH36_frames'
data_root_val = '/mnt/lustre21/DATAshare2/duanhaodong/Kinetics400/kinetics_400_val_SH36_frames'
ann_file_train = '/mnt/lustre21/DATAshare2/duanhaodong/Kinetics400/kinetics_train_list_hd320.txt'
ann_file_val = '/mnt/lustre21/DATAshare2/duanhaodong/Kinetics400/kinetics_val_list_hd320.txt'
ann_file_test = '/mnt/lustre21/DATAshare2/duanhaodong/Kinetics400/kinetics_val_list_hd320.txt'

mc_cfg = dict(
    server_list_cfg = '/mnt/lustre/share/memcached_client/server_list.conf',
    client_cfg = '/mnt/lustre/share/memcached_client/client.conf',
    sys_path= '/mnt/lustre/share/pymc/py3'
)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1),
    # dict(type='RawFrameDecode'),
    dict(type='RawFrameDecode', io_backend='memcached', **mc_cfg),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=2,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode', io_backend='memcached', **mc_cfg),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=2,
        num_clips=10,
        test_mode=True),
    dict(type='RawFrameDecode', io_backend='memcached', **mc_cfg),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=8,
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
evaluation = dict(
    interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'])

# optimizer
# optimizer = dict(
#     type='SGD', lr=0.1, momentum=0.9,
#     weight_decay=0.0001)  # this lr is used for 8 gpus
optimizer = dict(
    type='SGD', lr=0.2, momentum=0.9,
    weight_decay=0.0001)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=34)
total_epochs = 256

# precise_BN
precise_bn=dict(num_iters=200, interval=1)

# add checkpoint 
# load_from = None
# load_from = '/mnt/lustre/liguankai/resume/slowfast_r50_4x16x1_256e_kinetics400_rgb_20200704-bcde7ed7.pth'
# resume_from = '/mnt/lustre/liguankai/work_dir/sl_prebn/latest.pth'

# runtime settings
checkpoint_config = dict(interval=4)
work_dir = './work_dirs/slowfast_r50_3d_4x16x1_256e_kinetics400_rgb_test'
find_unused_parameters = False
