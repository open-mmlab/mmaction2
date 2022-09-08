# This config is used to generate long-term feature bank.
_base_ = [
    '../../_base_/default_runtime.py', '../_base_/models/slowonly_r50.py'
]

# model settings
lfb_prefix_path = 'data/ava/lfb_half'
dataset_mode = 'val'  # ['train', 'val', 'test']

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

file_client_args = dict(
    io_backend='petrel',
    path_mapping=dict({'data/ava': 's3://openmmlab/datasets/action/ava'}))

infer_pipeline = [
    dict(
        type='SampleAVAFrames', clip_len=4, frame_interval=16, test_mode=True),
    dict(type='RawFrameDecode', **file_client_args),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='FormatShape', input_format='NCTHW', collapse=True),
    dict(type='PackActionInputs')
]

test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_infer,
        exclude_file=exclude_file_infer,
        pipeline=infer_pipeline,
        label_file=label_file,
        proposal_file=proposal_file_infer,
        data_prefix=dict(img=data_root),
        person_det_score_thr=0.9,
        test_mode=True))

test_evaluator = dict(
    type='AVAMetric',
    ann_file=ann_file_infer,
    label_file=label_file,
    exclude_file=exclude_file_infer)

test_cfg = dict(type='TestLoop')
