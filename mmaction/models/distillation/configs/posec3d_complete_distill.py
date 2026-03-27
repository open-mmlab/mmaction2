_base_ = ['./posec3d_k400.py']

# Teacher model config and checkpoint
teacher_config = 'configs/skeleton/posec3d/posec3d_k400.py'
teacher_checkpoint = 'checkpoints/posec3d_k400-73b07ecd.pth'

# Student model will be much smaller than teacher
model = dict(
    type='ProgressiveDistillPoseRecognizer',
    backbone=dict(
        type='PoseC3DStudentBackbone',
        depth=18,  # ResNet18 vs teacher's ResNet50
        in_channels=17,
        base_channels=32,  # Half the channels of teacher
        num_stages=4,
        out_indices=(3,),
        stage_blocks=(2, 2, 2, 2),
        conv1_stride_s=1,
        pool1_stride_s=1,
        inflate=(0, 1, 1, 1),
        spatial_strides=(1, 2, 2, 2),
        temporal_strides=(1, 1, 1, 1)),
    cls_head=dict(
        type='PoseC3DDistillHead',
        in_channels=256,  # Reduced from 2048
        num_classes=400,
        loss_cls=dict(type='CrossEntropyLoss')),
    
    # Teacher configuration
    teacher_config=teacher_config,
    teacher_ckpt=teacher_checkpoint,
    
    # Feature distillation
    feature_dist_cfg=dict(
        type='FeatureDistillationLoss',
        student_channels=256,  # From student's last stage
        teacher_channels=2048, # From teacher's last stage
        distill_type='mse',
        transform_type='linear',
        weight=0.5,
    ),
    
    # Attention distillation
    attention_dist_cfg=dict(
        type='AttentionTransferLoss',
        beta=1.0,
        normalize=True,
    ),
    
    # Logit distillation with temperature scaling
    logit_dist_cfg=dict(
        type='KDLoss',
        temperature=4.0,
        alpha=0.5,
    ),
    
    # Progressive distillation strategy
    progressive_cfg=dict(
        phases=[
            dict(epochs=(0, 20), distill_types=['feature']),  # Start with feature matching
            dict(epochs=(20, 40), distill_types=['feature', 'attention']),  # Add attention
            dict(epochs=(40, -1), distill_types=['feature', 'attention', 'logit']),  # Full distillation
        ]
    ),
)

# Add custom hooks for distillation process
custom_hooks = [
    dict(
        type='DistillationHook',
        # Alpha scheduling: gradually increase the weight of KD loss
        alpha_scheduler=lambda epoch, max_epochs: min(0.1 + epoch / max_epochs, 0.5),
        # Temperature scheduling: gradually reduce temperature
        temp_scheduler=lambda epoch, max_epochs: max(4.0 - 3.0 * epoch / max_epochs, 1.0),
    ),
]

# Optimizer: smaller LR for distillation
optimizer = dict(
    type='SGD',
    lr=0.02,  # Reduced from standard training
    momentum=0.9,
    weight_decay=0.0001,
    nesterov=True,
    paramwise_cfg=dict(
        # Apply different LR to backbone vs. other components
        custom_keys={
            'backbone': dict(lr_mult=0.1),  # Lower LR for backbone
            'cls_head': dict(lr_mult=1.0),  # Normal LR for classification head
        }
    )
)

# Learning rate config
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_iters=2000,
    warmup_ratio=0.1
)

# Runtime settings
total_epochs = 100  # Longer training for distillation
evaluation = dict(interval=5)  # Evaluate more frequently
checkpoint_config = dict(interval=5)  # Save checkpoint more frequently
log_config = dict(interval=20)

# Working directory
work_dir = './work_dirs/posec3d_distill_complete/'