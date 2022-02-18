# This config is used to generate long-term feature bank.
_base_ = ['../_base_/models/slowonly_r50.py']

# model settings
lfb_prefix_path = 'data/ava/lfb_half'
dataset_mode = 'train'  # ['train', 'val', 'test']

model = dict(
    roi_head=dict(
        shared_head=dict(
            type='LFBInferHead',
            lfb_prefix_path=lfb_prefix_path,
            dataset_mode=dataset_mode,
            use_half_precision=True)))

# dataset settings
dataset_type = 'AVADataset'
data_root = 'data/ava/rawframes'
anno_root = 'data/ava/annotations'

ann_file_infer = f'{anno_root}/ava_{dataset_mode}_v2.1.csv'

exclude_file_infer = (
    f'{anno_root}/ava_{dataset_mode}_excluded_timestamps_v2.1.csv')

label_file = f'{anno_root}/ava_action_list_v2.1_for_activitynet_2018.pbtxt'

proposal_file_infer = (
    f'{anno_root}/ava_dense_proposals_{dataset_mode}.FAIR.recall_93.9.pkl')

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

infer_pipeline = [
    dict(
        type='SampleAVAFrames', clip_len=4, frame_interval=16, test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW', collapse=True),
    # Rename is needed to use mmdet detectors
    dict(type='Rename', mapping=dict(imgs='img')),
    dict(type='ToTensor', keys=['img', 'proposals']),
    dict(type='ToDataContainer', fields=[dict(key='proposals', stack=False)]),
    dict(
        type='Collect',
        keys=['img', 'proposals'],
        meta_keys=['scores', 'img_shape', 'img_key'],
        nested=True)
]

data = dict(
    videos_per_gpu=1,
    workers_per_gpu=2,
    test=dict(
        type=dataset_type,
        ann_file=ann_file_infer,
        exclude_file=exclude_file_infer,
        pipeline=infer_pipeline,
        label_file=label_file,
        proposal_file=proposal_file_infer,
        person_det_score_thr=0.9,
        data_prefix=data_root))

dist_params = dict(backend='nccl')
