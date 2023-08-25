_base_ = ['../../_base_/default_runtime.py']

video_root = 'data/msrvtt/msrvtt_2fps_224'
anno_file_train = 'data/msrvtt/anno_downstream/msrvtt_ret_train9k.json'
anno_file_test = 'data/msrvtt/anno_downstream/msrvtt_ret_test1k.json'
pretrained_ckpt_path = 'checkpoints/5M-pretrain.pth'

# model settings
model = dict(
    type='VindLURet',
    pretrained_ckpt=pretrained_ckpt_path,
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[128],
        std=[128],
        format_shape='NCTHW'),
    tokenizer=dict(
        type='BertTokenizer',
        pretrained_model_name_or_path='bert-base-uncased'),
    vision_encoder=dict(
        type='beit',
        tem_config=dict(
            num_frames=12,
            temporal_model_block='timesformer',
            temporal_model_position='last',
            temporal_model_config=dict(input_dim=768),
            use_temporal_position_embedding=True),
        pretrained_model_name_or_path=
        'microsoft/beit-base-patch16-224-pt22k-ft22k',
        encoder_width=768,
        add_ln=True),
    text_encoder=dict(
        type='XBertModel',
        pretrained_model_name_or_path='bert-base-uncased',
        encoder_width=768,
        fusion_layer=9,
        add_pooling_layer=False),
    proj_dim=256,
    temperature=0.07,
    evaluate=True,
    max_txt_len=32,
    topk=128,
    gradient_checkpointing=True)

file_client_args = dict(io_backend='disk')
train_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=12),
    dict(type='DecordDecode'),
    dict(type='RandomResizedCrop', area_range=(0.5, 1.0)),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCHW'),
    dict(
        type='PackActionInputs',
        algorithm_keys=(
            'text',
            'gt_video_id',
            'gt_text_id',))
]

val_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=12,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='FormatShape', input_format='NCHW'),
    dict(
        type='PackActionInputs',
        algorithm_keys=(
            'text',
            'gt_video_id',
            'gt_text_id',
        ))
]

test_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=12,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='FormatShape', input_format='NCHW'),
    dict(
        type='PackActionInputs',
        algorithm_keys=(
            'text',
            'gt_video_id',
            'gt_text_id',
        ))
]

dataset_type = 'MSRVTT_Ret'

train_dataloader = dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=anno_file_train,
        pipeline=train_pipeline,
        data_prefix=dict(video=video_root),
    ))

val_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=anno_file_test,
        pipeline=test_pipeline,
        data_prefix=dict(video=video_root),
    ))

test_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=anno_file_test,
        pipeline=test_pipeline,
        data_prefix=dict(video=video_root),
    ))

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=5, val_begin=1, val_interval=1)
val_cfg = dict(type='RetrievalValLoop')
test_cfg = dict(type='RetrievalTestLoop')

val_evaluator = dict(type='RetrievalRecall',  topk=(1, 5, 10))
test_evaluator = dict(type='RetrievalRecall',  topk=(1, 5, 10))

param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=5,
        eta_min_ratio=0.01,
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True)
]

optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-5, weight_decay=0.02),
    paramwise_cfg=dict(bypass_duplicate=True, norm_decay_mult=0.0, bias_decay_mult=0.0),
    clip_grad=dict(max_norm=50, norm_type=2),
    )

model_wrapper_cfg=dict(type='MMDistributedDataParallel', static_graph=True)


default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, save_best="t2i/retrieval/Recall@1", rule='greater'),
    logger=dict(type='LoggerHook', interval=20, ignore_last=False))

auto_scale_lr = dict(enable=True, base_batch_size=128)

find_unused_parameters=True

custom_hooks = [dict(type='EmptyCacheHook', after_epoch=True)]