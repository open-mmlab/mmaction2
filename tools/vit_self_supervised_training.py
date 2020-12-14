import torch
from byol_pytorch import BYOL

from mmaction.datasets import VideoDataset, build_dataloader
from mmaction.models.backbones import ViT

model = ViT(
    image_size=256,
    patch_size=32,
    num_classes=212,
    dim=1024,
    depth=6,
    heads=8,
    mlp_dim=2048)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='PyAVInit'),
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=3),
    dict(type='PyAVDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='MultiScaleCrop',
        input_size=256,
        scales=(1, 0.875, 0.75, 0.66),
        random_crop=False,
        max_wh_scale_gap=1),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg)
]

frame_ann_file = 'data/kinetics400/kinetics400_train_list_videos.txt'
ds = VideoDataset(
    frame_ann_file, train_pipeline, data_prefix='data/kinetics400/')
data_loader = build_dataloader(
    ds, videos_per_gpu=20, workers_per_gpu=4, num_gpus=1, dist=False)

learner = BYOL(model, image_size=256, hidden_layer='to_cls_token')

opt = torch.optim.Adam(learner.parameters(), lr=3e-4)

for _ in range(10):
    for i, data_batch in enumerate(data_loader):
        imgs = data_batch.get('imgs', None)
        loss = learner(imgs)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i % 10 == 0:
            print(f'{i}/{len(data_loader)}: {loss.data}')
        learner.update_moving_average(
        )  # update moving average of target encoder
        torch.save(model.state_dict(), './pretrained-net.pth')
