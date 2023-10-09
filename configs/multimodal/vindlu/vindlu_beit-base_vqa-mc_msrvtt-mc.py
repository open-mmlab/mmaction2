_base_ = ['../../_base_/default_runtime.py']

video_root = 'data/msrvtt/videos_2fps_224'
anno_file_test = 'data/msrvtt/annotations/msrvtt_mc_test.json'

# model settings
model = dict(
    type='VindLURetrievalMC',
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[128],
        std=[128],
        format_shape='NCTHW'),
    tokenizer=dict(
        type='VindLUTokenizer',
        pretrained_model_name_or_path='bert-base-uncased'),
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
    max_txt_len=32,
    gradient_checkpointing=True)

file_client_args = dict(io_backend='disk')

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
    dict(type='PackActionInputs', algorithm_keys=('caption_options', ))
]

dataset_type = 'MSRVTTVQAMC'

test_dataloader = dict(
    batch_size=32,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=anno_file_test,
        pipeline=test_pipeline,
        data_prefix=dict(video=video_root),
    ))

test_evaluator = dict(type='VQAMCACC')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=20, ignore_last=False), )
