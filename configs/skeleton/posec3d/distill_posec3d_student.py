# configs/skeleton/posec3d/distill_posec3d_student.py

_base_ = ['../../_base_/default_runtime.py']

# model settings
model = dict(
    type='DistillPoseC3D',
    backbone=dict(
        type='ResNet3dSlowOnly',  # Student backbone (smaller)
        in_channels=17,  # Number of keypoints
        base_channels=32,  # Reduced from original (64)
        num_stages=3,
        out_indices=(2, ),
        stage_blocks=(4, 6, 3),
        conv1_stride_s=1,
        pool1_stride_s=1,
        inflate=(0, 1, 1),
        spatial_strides=(2, 2, 2),
        temporal_strides=(1, 1, 2),
        dilations=(1, 1, 1)),
    teacher_backbone=dict(
        type='ResNet3dSlowOnly',  # Teacher backbone (larger)
        in_channels=17,
        base_channels=64,  # Original size
        num_stages=4,
        out_indices=(3, ),
        stage_blocks=(3, 4, 6, 3),
        conv1_stride_s=1,
        pool1_stride_s=1,
        inflate=(0, 1, 1, 1),
        spatial_strides=(2, 2, 2, 2),
        temporal_strides=(1, 1, 2, 2),
        dilations=(1, 1, 1, 1)),
    teacher_checkpoint='checkpoints/posec3d_k400.pth',  # Path to teacher model
    cls_head=dict(
        type='I3DHead',
        in_channels=512,  # Match student backbone output
        num_classes=60,  # NTU60 has 60 classes
        spatial_type='avg',
        dropout_ratio=0.5),
    teacher_cls_head=dict(
        type='I3DHead',
        in_channels=512,  # Match teacher backbone output
        num_classes=60,
        spatial_type='avg',
        dropout_ratio=0.5),
    train_cfg=dict(
        distill_loss=dict(
            type='KLDivLoss',
            temperature=4.0,
            alpha=0.5  # Balance between distillation and CE loss
        )
    ),
    test_cfg=dict(average_clips='prob')
)

# dataset settings
dataset_type = 'PoseDataset'
ann_file_train = '/home/minhtranh/works/Project/Rainscales/Lying_detection/mmaction2/data/skeleton/ntu60_2d/ntu60_2d_train.pkl'
ann_file_val = '/home/minhtranh/works/Project/Rainscales/Lying_detection/mmaction2/data/skeleton/ntu60_2d/ntu60_2d_val.pkl'
train_pipeline = [
    dict(type='UniformSampleFrames', clip_len=48),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(-1, 64)),
    dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),
    dict(type='Resize', scale=(56, 56), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='GeneratePoseTarget',
         sigma=0.6,
         use_score=True,
         with_kp=True,
         with_limb=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='UniformSampleFrames', clip_len=48, num_clips=1),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(56, 56), keep_ratio=False),
    dict(type='GeneratePoseTarget',
         sigma=0.6,
         use_score=True,
         with_kp=True,
         with_limb=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = val_pipeline
data = dict(
    videos_per_gpu=16,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix='',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix='',
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix='',
        pipeline=test_pipeline))

# optimizer
optimizer = dict(
    type='SGD',
    lr=0.01,  # Lower learning rate for distillation
    momentum=0.9,
    weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=5)
total_epochs = 10

# runtime settings
checkpoint_config = dict(interval=5)
evaluation = dict(interval=5, metrics=[
                  'top_k_accuracy', 'mean_class_accuracy'])
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])

# Make sure the teacher model checkpoint exists
# If you don't have it, you can download it or use a different checkpoint
# You can also set this to None and the model will use random weights (not recommended)
find_unused_parameters = True  # Important for distillation training
