# model settings
lfb_prefix_path = 'data/ava/lfb'
dataset_mode = 'train'  # dataset_mode are in ['train', 'val', 'test']

model = dict(
    type='FastRCNN',
    backbone=dict(
        type='ResNet3dSlowOnly',
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
        shared_head=dict(
            type='LFBInferHead',
            lfb_prefix_path=lfb_prefix_path,
            dataset_mode=dataset_mode),
        bbox_head=dict(
            type='BBoxHeadAVA',
            in_channels=2048,
            num_classes=81,
            multilabel=True,
            dropout_ratio=0.5)))
# model testing settings
test_cfg = dict(rcnn=dict(action_thr=0.00))

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

mc_cfg = dict(
    server_list_cfg='/mnt/lustre/share/memcached_client/server_list.conf',
    client_cfg='/mnt/lustre/share/memcached_client/client.conf',
    sys_path='/mnt/lustre/share/pymc/py3')

infer_pipeline = [
    dict(type='SampleAVAFrames', clip_len=4, frame_interval=16),
    dict(type='RawFrameDecode', io_backend='memcached', **mc_cfg),
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
    videos_per_gpu=16,
    workers_per_gpu=4,
    test_dataloader=dict(videos_per_gpu=1),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_infer,
        exclude_file=exclude_file_infer,
        filename_tmpl='img_{:06d}.jpg',
        pipeline=infer_pipeline,
        label_file=label_file,
        proposal_file=proposal_file_infer,
        person_det_score_thr=0.9,
        data_prefix=data_root))
