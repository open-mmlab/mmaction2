_base_ = ['./lfb_avg_kinetics_pretrained_slowonly_r50_4x16x1_20e_ava_rgb.py']

max_num_sampled_feat = 5
window_size = 60
lfb_channels = 2048

model = dict(
    roi_head=dict(
        shared_head=dict(
            fbo_cfg=dict(
                type='non_local',
                st_feat_channels=2048,
                lt_feat_channels=lfb_channels,
                latent_channels=512,
                num_st_feat=1,
                num_lt_feat=window_size * max_num_sampled_feat,
                num_non_local_layers=2,
                st_feat_dropout_ratio=0.2,
                lt_feat_dropout_ratio=0.2,
                pre_activate=True)),
        bbox_head=dict(in_channels=2560)))