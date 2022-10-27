_base_ = '2sagcn_4xb16-80e_ntu60-xsub-keypoint-3d.py'

dataset_type = 'PoseDataset'
ann_file_train = 'data/ntu/nturgb+d_skeletons_60_3d/xsub/train.pkl'
ann_file_val = 'data/ntu/nturgb+d_skeletons_60_3d/xsub/val.pkl'
train_pipeline = [
    dict(type='PaddingWithLoop', clip_len=300),
    dict(type='PoseDecode'),
    dict(type='JointToBone'),
    dict(type='FormatGCNInput', input_format='NCTVM'),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(type='PaddingWithLoop', clip_len=300),
    dict(type='PoseDecode'),
    dict(type='JointToBone'),
    dict(type='FormatGCNInput', input_format='NCTVM'),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type, ann_file=ann_file_train, pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type, ann_file=ann_file_val, pipeline=train_pipeline))
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type, ann_file=ann_file_val, pipeline=train_pipeline))
