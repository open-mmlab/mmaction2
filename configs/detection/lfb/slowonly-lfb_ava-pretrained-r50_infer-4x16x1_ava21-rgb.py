# This config is used to generate long-term feature bank.
_base_ = ['../../_base_/default_runtime.py']

# model settings
lfb_prefix_path = 'data/ava/lfb_half'
dataset_mode = 'val'  # ['train', 'val', 'test']

url = ('https://download.openmmlab.com/mmaction/v1.0/recognition/slowonly/'
       'slowonly_imagenet-pretrained-r50_8xb16-4x16x1-steplr-150e_kinetics400-'
       'rgb/slowonly_imagenet-pretrained-r50_8xb16-4x16x1-steplr-150e_'
       'kinetics400-rgb_20220901-e7b65fad.pth')

model = dict(
    type='FastRCNN',
    _scope_='mmdet',
    init_cfg=dict(type='Pretrained', checkpoint=url),
    backbone=dict(
        type='mmaction.ResNet3dSlowOnly',
        depth=50,
        pretrained=None,
        pretrained2d=False,
        lateral=False,
        num_stages=4,
        conv1_kernel=(1, 7, 7),
        conv1_stride_t=1,
        pool1_stride_t=1,
        spatial_strides=(1, 2, 2, 1)),
    roi_head=dict(
        type='AVARoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor3D',
            roi_layer_type='RoIAlign',
            output_size=8,
            with_temporal_pool=True),
        bbox_head=dict(
            type='BBoxHeadAVA',
            background_class=True,
            in_channels=2048,
            num_classes=81,
            multilabel=True,
            dropout_ratio=0.5),
        shared_head=dict(
            type='LFBInferHead',
            lfb_prefix_path=lfb_prefix_path,
            dataset_mode=dataset_mode,
            use_half_precision=True)),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        _scope_='mmaction',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCTHW'),
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssignerAVA',
                pos_iou_thr=0.9,
                neg_iou_thr=0.9,
                min_pos_iou=0.9),
            sampler=dict(
                type='RandomSampler',
                num=32,
                pos_fraction=1,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=1.0)),
    test_cfg=dict(rcnn=None))

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
