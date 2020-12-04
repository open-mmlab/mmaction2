import torch

from mmaction.models import build_localizer


def test_tem():
    model_cfg = dict(
        type='TEM',
        temporal_dim=100,
        boundary_ratio=0.1,
        tem_feat_dim=400,
        tem_hidden_dim=512,
        tem_match_threshold=0.5)

    localizer_tem = build_localizer(model_cfg)
    raw_feature = torch.rand(8, 400, 100)
    gt_bbox = torch.Tensor([[[1.0, 3.0], [3.0, 5.0]]] * 8)
    losses = localizer_tem(raw_feature, gt_bbox)
    assert isinstance(losses, dict)

    # Test forward test
    video_meta = [{'video_name': 'v_test'}]
    with torch.no_grad():
        for one_raw_feature in raw_feature:
            one_raw_feature = one_raw_feature.reshape(1, 400, 100)
            localizer_tem(
                one_raw_feature, video_meta=video_meta, return_loss=False)
