_base_ = 'mmaction::recognition/tsm/tsm_imagenet-pretrained-mobilenetv2_8xb16-1x1x8-100e_kinetics400-rgb.py'  # noqa: E501

teacher_ckpt = 'https://download.openmmlab.com/mmaction/v1.0/recognition/tsm/tsm_imagenet-pretrained-r50_8xb16-1x1x8-50e_kinetics400-rgb/tsm_imagenet-pretrained-r50_8xb16-1x1x8-50e_kinetics400-rgb_20220831-64d69186.pth'  # noqa: E501
model = dict(
    _delete_=True,
    _scope_='mmrazor',
    type='SingleTeacherDistill',
    architecture=dict(
        cfg_path=  # noqa: E251
        'mmaction::recognition/tsm/tsm_imagenet-pretrained-mobilenetv2_8xb16-1x1x8-100e_kinetics400-rgb.py',  # noqa: E501
        pretrained=False),
    teacher=dict(
        cfg_path=  # noqa: E251
        'mmaction::recognition/tsm/tsm_imagenet-pretrained-r50_8xb16-1x1x8-50e_kinetics400-rgb.py',  # noqa: E501
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
                loss_weight=1,
            )),
        loss_forward_mappings=dict(
            loss_dist=dict(
                logits_S=dict(from_student=True, recorder='logits'),
                logits_T=dict(from_student=False, recorder='logits')))))

val_cfg = dict(_delete_=True, type='mmrazor.SingleTeacherDistillValLoop')
