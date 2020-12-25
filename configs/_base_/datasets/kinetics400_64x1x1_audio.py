# dataset settings
dataset_type = 'AudioDataset'
data_root = 'data/kinetics400/audios'
data_root_val = 'data/kinetics400/audios'
ann_file_train = 'data/kinetics400/kinetics400_train_list_audio.txt'
ann_file_val = 'data/kinetics400/kinetics400_val_list_audio.txt'
ann_file_test = 'data/kinetics400/kinetics400_val_list_audio.txt'
train_pipeline = [
    dict(type='AudioDecodeInit'),
    dict(type='SampleFrames', clip_len=64, frame_interval=1, num_clips=1),
    dict(type='AudioDecode'),
    dict(type='AudioAmplify', ratio=1.5),
    dict(type='MelLogSpectrogram'),
    dict(type='FormatAudioShape', input_format='NCTF'),
    dict(type='Collect', keys=['audios', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['audios'])
]
val_pipeline = [
    dict(type='AudioDecodeInit'),
    dict(
        type='SampleFrames',
        clip_len=64,
        frame_interval=1,
        num_clips=1,
        test_mode=True),
    dict(type='AudioDecode'),
    dict(type='AudioAmplify', ratio=1.5),
    dict(type='MelLogSpectrogram'),
    dict(type='FormatAudioShape', input_format='NCTF'),
    dict(type='Collect', keys=['audios', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['audios'])
]
test_pipeline = [
    dict(type='AudioDecodeInit'),
    dict(
        type='SampleFrames',
        clip_len=64,
        frame_interval=1,
        num_clips=1,
        test_mode=True),
    dict(type='AudioDecodeInit'),
    dict(type='AudioAmplify', ratio=1.5),
    dict(type='MelLogSpectrogram'),
    dict(type='FormatAudioShape', input_format='NCTF'),
    dict(type='Collect', keys=['audios', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['audios'])
]
data = dict(
    videos_per_gpu=320,
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
    interval=5, metrics=['top_k_accuracy', 'mean_class_accuracy'])
