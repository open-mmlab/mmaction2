_base_ = ['../../_base_/default_runtime.py']

video_root = 'data/msrvtt/msrvtt_2fps_224'
anno_file_test = 'data/msrvtt/anno_downstream/msrvtt_qa_test.json'
answer_list_file = 'data/msrvtt/anno_downstream/msrvtt_qa_answer_list.json'
vision_backbone_name = 'microsoft/beit-base-patch16-224-pt22k-ft22k'
text_backbone_config = 'configs/multimodal/vindlu/config_bert.json'

# model settings
model = dict(
    type='VindLU_QA',
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[128],
        std=[128],
        format_shape='NCTHW'),
    tokenizer=dict(
        max_question_len=25,
        max_answer_len=5,
        eos='[SEP]',
    ),
    vision_backbone=dict(
        type='beit',
        temporal_modeling={
            'num_frames': 12,
            'temporal_model_block': 'timesformer',
            'temporal_model_position': 'last',
            'temporal_model_config': {
                'input_dim': 768
            },
            'use_temporal_position_embedding': True,
        },
        pretrained=vision_backbone_name,
        pretrained_path=
        'work_dirs/ft_12frm-pt_webvid_cc3m_8x64-qa_msrvtt/ckpt_best.pth',
        d_model=768,
        add_ln=True,
        image_res=224,
    ),
    text_backbone=dict(
        type='bert_base',
        pretrained='bert-base-uncased',
        d_model=768,
        fusion_layer=9,
        config=text_backbone_config,
        multimodal=True,
        is_pretrain=False,
    ),
    proj_dim=256,
    temperature=0.07,
    has_decoder=True,
    evaluate=True,
    gradient_checkpointing=True,
    k=128,
    answer_list_path=answer_list_file)

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
    dict(
        type='PackActionInputs',
        algorithm_keys=(
            'question',
            'gt_answer',
            'question_id',
        ))
]

dataset_type = 'MSRVTT_VQA'

test_dataloader = dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=anno_file_test,
        pipeline=test_pipeline,
        k=128,
        test_mode=True,
        data_prefix=dict(video=video_root),
    ))

test_evaluator = dict(type='VQAAcc')
test_cfg = dict(type='TestLoop')
