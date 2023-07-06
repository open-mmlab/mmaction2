_base_ = 'mmaction::recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb.py'  # noqa: E501

teacher_ckpt = 'https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-swin-transformer_32xb8-1x1x8-50e_kinetics400-rgb/tsn_imagenet-pretrained-swin-transformer_32xb8-1x1x8-50e_kinetics400-rgb_20230530-428f0064.pth'  # noqa: E501

model = dict(
    _delete_=True,
    _scope_='mmrazor',
    type='SingleTeacherDistill',
    architecture=dict(
        cfg_path=  # noqa: E251
        'mmaction::recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb.py',  # noqa: E501
        backbone=dict(pretrained=False),
        pretrained=False),
    teacher=dict(
        cfg_path=  # noqa: E251
        'mmaction::recognition/tsn/custom_backbones/tsn_imagenet-pretrained-swin-transformer_32xb8-1x1x8-50e_kinetics400-rgb.py',  # noqa: E501
        pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type='ConfigurableDistiller',
        student_recorders=dict(
            logits=dict(type='ModuleOutputs', source='cls_head.fc_cls')),
        teacher_recorders=dict(
            logits=dict(type='ModuleOutputs', source='cls_head.fc_cls')),
        distill_losses=dict(
            loss_dist=dict(
                type='DISTLoss',
                inter_loss_weight=1.0,
                intra_loss_weight=1.0,
                tau=1,
                loss_weight=4,
            )),
        loss_forward_mappings=dict(
            loss_dist=dict(
                logits_S=dict(from_student=True, recorder='logits'),
                logits_T=dict(from_student=False, recorder='logits')))))

val_cfg = dict(_delete_=True, type='mmrazor.SingleTeacherDistillValLoop')
