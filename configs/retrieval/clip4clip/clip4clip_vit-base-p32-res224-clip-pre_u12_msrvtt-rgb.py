_base_ = '../../_base_/default_runtime.py'


model = dict(
    type='CLIPSimilarity',
    clip_arch='ViT-B/32',
    to_float32=True,
    data_preprocessor=dict(
        type='MultiModalDataPreprocessor',
        preprocessors=dict(
            imgs=dict(
                type='ActionDataPreprocessor',
                mean=[122.771, 116.746, 104.093],
                std=[68.500, 66.632, 70.323],
                format_shape='NCHW'),
            text=dict(type='ActionDataPreprocessor', to_float32=False))))

dataset_type = 'VideoTextDataset'
data_root = 'data/video_retrieval/msrvtt'
file_client_args = dict(io_backend='disk')
train_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='UniformSample', clip_len=12, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='CLIPTokenize'),
    dict(type='PackActionInputs', collect_keys=('imgs', 'text'))
]
val_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='UniformSample', clip_len=12, num_clips=1, test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='CLIPTokenize'),
    dict(type='PackActionInputs', collect_keys=('imgs', 'text'))
]
test_pipeline = val_pipeline

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file='train_9k.json',
        data_root=data_root,
        data_prefix=dict(video='videos'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file='test_JSFUSION.json',
        data_root=data_root,
        data_prefix=dict(video='videos'),
        pipeline=val_pipeline,
        test_mode=True))
test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file='test_JSFUSION.json',
        data_root=data_root,
        data_prefix=dict(video='videos'),
        pipeline=test_pipeline,
        test_mode=True))

val_evaluator = dict(type='RetrievalMetric')
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=5, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=1e-4, momentum=0.9, weight_decay=1e-4),
    clip_grad=dict(max_norm=1.0, norm_type=2))

default_hooks = dict(checkpoint=dict(interval=3, max_keep_ckpts=3))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (16 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=128)