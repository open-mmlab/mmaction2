# model settings
model = dict(
    type='BMN',
    temporal_dim=100,
    boundary_ratio=0.5,
    num_samples=32,
    num_samples_per_bin=3,
    feat_dim=400,
    soft_nms_alpha=0.4,
    soft_nms_low_threshold=0.5,
    soft_nms_high_threshold=0.9,
    post_process_top_k=100)
