# model settings
model = dict(
    type='RecognizerAudio',
    backbone=dict(type='ResNet', depth=18, in_channels=1, norm_eval=False),
    cls_head=dict(
        type='TSNHeadAudio',
        num_classes=212,
        in_channels=512,
        dropout_ratio=0.4,
        init_std=0.01))
# model training and testing settings
train_cfg = None
test_cfg = dict(average_clips=None)
# dataset settings
dataset_type = 'AudioDataset'
data_root = 'data/ugc/audios'
data_root_val = 'data/ugc/audios'
ann_file_train = 'data/ugc/ugc_train_list_audio.txt'
ann_file_val = 'data/ugc/ugc_val_list_audio.txt'
ann_file_test = 'data/ugc/ugc_val_list_audio.txt'
mc_cfg = dict(
    server_list_cfg='/mnt/lustre/share/memcached_client/server_list.conf',
    client_cfg='/mnt/lustre/share/memcached_client/client.conf',
    sys_path='/mnt/lustre/share/pymc/py3')
train_pipeline = [
    dict(type='AudioDecodeInit', io_backend='memcached', **mc_cfg),
    dict(type='SampleFrames', clip_len=64, frame_interval=1, num_clips=1),
    dict(type='AudioDecode'),
    # dict(type='AudioAmplify', ratio=1.5),
    dict(type='MelSpectrogram'),
    dict(type='FormatAudioShape', input_format='NCTF'),
    dict(type='Collect', keys=['audios', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['audios', 'label'])
]
val_pipeline = [
    dict(type='AudioDecodeInit', io_backend='memcached', **mc_cfg),
    dict(
        type='SampleFrames',
        clip_len=64,
        frame_interval=1,
        num_clips=1,
        test_mode=True),
    dict(type='AudioDecode'),
    # dict(type='AudioAmplify'),
    dict(type='MelSpectrogram'),
    dict(type='FormatAudioShape', input_format='NCTF'),
    dict(type='Collect', keys=['audios', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['audios', 'label'])
]
test_pipeline = [
    dict(type='AudioDecodeInit', io_backend='memcached', **mc_cfg),
    dict(
        type='SampleFrames',
        clip_len=64,
        frame_interval=1,
        num_clips=1,
        test_mode=True),
    dict(type='AudioDecode'),
    # dict(type='AudioAmplify'),
    dict(type='MelSpectrogram'),
    dict(type='FormatAudioShape', input_format='NCTF'),
    dict(type='Collect', keys=['audios', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['audios', 'label'])
]
data = dict(
    videos_per_gpu=16,
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
    type='SGD', lr=0.01, momentum=0.9,
    weight_decay=0.0001)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[10, 15])
total_epochs = 20
checkpoint_config = dict(interval=5)
evaluation = dict(
    interval=5, metrics=['top_k_accuracy', 'mean_class_accuracy'], topk=(1, 5))
log_config = dict(
    interval=2,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
    ])
# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/tsn_r18_128x1x1_ugc_audio/'
load_from = None
resume_from = None
workflow = [('train', 1)]
