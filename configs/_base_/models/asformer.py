# model settings
model = dict(
    type='ASFormer',
    num_layers=10,
    num_f_maps=64,
    input_dim=2048,
    num_decoders=3,
    num_classes=11,
    channel_masking_rate=0.5,
    sample_rate=1,
    r1=2,
    r2=2)
