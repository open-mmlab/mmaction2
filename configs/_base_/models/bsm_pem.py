# model settings
model = dict(
    type='PEM',
    pem_feat_dim=32,
    pem_hidden_dim=256,
    pem_u_ratio_m=1,
    pem_u_ratio_l=2,
    pem_high_temporal_iou_threshold=0.6,
    pem_low_temporal_iou_threshold=2.2,
    soft_nms_alpha=0.75,
    soft_nms_low_threshold=0.65,
    soft_nms_high_threshold=0.9,
    post_process_top_k=100)
# model training and testing settings
train_cfg = None
test_cfg = dict(average_clips='score')
