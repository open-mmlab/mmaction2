_base_ = ['slowonly_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb.py']

model = dict(
    backbone=dict(
        resample_rate=4,
        speed_ratio=4,
        slow_pathway=dict(fusion_kernel=7),
        prtrained=(
            'https://download.openmmlab.com/mmaction/recognition/slowfast/'
            'slowfast_r50_8x8x1_256e_kinetics400_rgb/'
            'slowfast_r50_8x8x1_256e_kinetics400_rgb_20200716-73547d2b.pth')))

dataset_type = 'AVADataset'
data_root = 'data/ava/rawframes'
anno_root = 'data/ava/annotations'

ann_file_train = f'{anno_root}/ava_train_v2.1.csv'
exclude_file_train = f'{anno_root}/ava_train_excluded_timestamps_v2.1.csv'
label_file = f'{anno_root}/ava_action_list_v2.1_for_activitynet_2018.pbtxt'

proposal_file_train = (f'{anno_root}/ava_dense_proposals_train.FAIR.'
                       'recall_93.9.pkl')

train_pipeline = [
    dict(type='SampleAVAFrames', clip_len=32, frame_interval=2),
    dict(type='RawFrameDecode'),
    dict(type='RandomRescale', scale_range=(256, 320)),
    dict(type='RandomCrop', size=256),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW', collapse=True),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        exclude_file=exclude_file_train,
        pipeline=train_pipeline,
        label_file=label_file,
        proposal_file=proposal_file_train,
        data_prefix=dict(img=data_root)))

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.00001),
    clip_grad=dict(max_norm=40, norm_type=2))
