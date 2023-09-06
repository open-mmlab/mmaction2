_base_ = ['../../_base_/default_runtime.py']

video_root = 'data/msrvtt/videos_2fps_224'
anno_file_train = 'data/msrvtt/annotations/msrvtt_qa_train.json'
anno_file_val = 'data/msrvtt/annotations/msrvtt_qa_val.json'
anno_file_test = 'data/msrvtt/annotations/msrvtt_qa_test.json'
answer_list_file = 'data/msrvtt/annotations/msrvtt_qa_answer_list.json'
pretrained_ckpt_url = 'https://download.openmmlab.com/mmaction/v1.0/multimodal/vindlu/vindlu_c5m_pretrain.pth'  # noqa: E501

# model settings
model = dict(
    type='VindLUVQA',
    init_cfg=dict(type='Pretrained', checkpoint=pretrained_ckpt_url),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[128],
        std=[128],
        format_shape='NCTHW'),
    tokenizer=dict(
        type='VindLUTokenizer',
        pretrained_model_name_or_path='bert-base-uncased',
    ),
    vision_encoder=dict(
        type='BeitModel3D',
        config='microsoft/beit-base-patch16-224-pt22k-ft22k',
        tem_config=dict(
            num_frames=12,
            temporal_model_block='timesformer',
            temporal_model_position='last',
            temporal_model_config=dict(input_dim=768),
            use_temporal_position_embedding=True),
        encoder_width=768,
        add_ln=True),
    text_encoder=dict(
        type='XBertModel',
        pretrained_model_name_or_path='bert-base-uncased',
        encoder_width=768,
        fusion_layer=9,
        add_pooling_layer=False),
    text_decoder=dict(
        type='BertDecoder',
        pretrained_model_name_or_path='bert-base-uncased',
        encoder_width=768,
        fusion_layer=0,
        num_hidden_layers=3,
        add_pooling_layer=True),
    proj_dim=256,
    temperature=0.07,
    max_question_len=25,
    max_answer_len=5,
    num_ans_candidates=128,
    gradient_checkpointing=True,
    answer_list_path=answer_list_file)

file_client_args = dict(io_backend='disk')

train_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=12,
        out_of_bound_opt='repeat_last'),
    dict(type='DecordDecode'),
    dict(type='RandomResizedCrop', area_range=(0.5, 1.0)),
    dict(
        type='Resize',
        scale=(224, 224),
        keep_ratio=False,
        interpolation='bicubic'),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCHW'),
    dict(
        type='PackActionInputs',
        algorithm_keys=(
            'question',
            'question_id',
            'gt_answer',
            'gt_answer_weight',
        ))
]

val_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=12,
        test_mode=True,
        out_of_bound_opt='repeat_last'),
    dict(type='DecordDecode'),
    dict(
        type='Resize',
        scale=(224, 224),
        keep_ratio=False,
        interpolation='bicubic'),
    dict(type='FormatShape', input_format='NCHW'),
    dict(
        type='PackActionInputs',
        algorithm_keys=(
            'question',
            'gt_answer',
            'question_id',
        ))
]

test_pipeline = val_pipeline

dataset_type = 'MSRVTTVQA'

train_dataloader = dict(
    batch_size=8,
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
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=anno_file_val,
        pipeline=val_pipeline,
        data_prefix=dict(video=video_root),
    ))

test_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=anno_file_test,
        pipeline=test_pipeline,
        data_prefix=dict(video=video_root),
    ))

val_evaluator = dict(type='VQAAcc')
test_evaluator = dict(type='VQAAcc')

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=10, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.01,
        by_epoch=True,
        begin=0,
        end=1,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=10,
        eta_min_ratio=0.01,
        by_epoch=True,
        begin=1,
        end=10,
        convert_to_iter_based=True)
]

optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-5, weight_decay=0.02),
    paramwise_cfg=dict(
        bypass_duplicate=True, norm_decay_mult=0.0, bias_decay_mult=0.0),
    clip_grad=dict(max_norm=50, norm_type=2),
)

model_wrapper_cfg = dict(type='MMDistributedDataParallel', static_graph=True)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=20, ignore_last=False))

auto_scale_lr = dict(enable=True, base_batch_size=32)

find_unused_parameters = True
