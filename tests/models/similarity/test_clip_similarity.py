# Copyright (c) OpenMMLab. All rights reserved.
import platform
from unittest.mock import MagicMock

import pytest
import torch

from mmaction.registry import MODELS
from mmaction.structures import ActionDataSample
from mmaction.testing import get_similarity_cfg
from mmaction.utils import register_all_modules


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_clip_similarity():
    register_all_modules()
    cfg = get_similarity_cfg(
        'clip4clip/'
        'clip4clip_vit-base-p32-res224-clip-pre_8xb16-u12-5e_msrvtt-9k-rgb.py')
    cfg.model.frozen_layers = -1  # no frozen layers
    model = MODELS.build(cfg.model)
    model.train()

    data_batch = {
        'inputs': {
            'imgs': [torch.randint(0, 256, (2, 3, 224, 224))],
            'text': [torch.randint(0, 49408, (77, ))]
        },
        'data_samples': [ActionDataSample()]
    }

    # test train_step
    optim_wrapper = MagicMock()
    loss_vars = model.train_step(data_batch, optim_wrapper)
    assert 'loss' in loss_vars
    assert 'sim_loss_v2t' in loss_vars
    assert 'sim_loss_t2v' in loss_vars
    optim_wrapper.update_params.assert_called_once()

    # test test_step
    with torch.no_grad():
        predictions = model.test_step(data_batch)
    features = predictions[0].features
    assert len(predictions) == 1
    assert features.video_feature.size() == (512, )
    assert features.text_feature.size() == (512, )

    # test frozen layers
    def check_frozen_layers(mdl, frozen_layers):
        if frozen_layers >= 0:
            top_layers = [
                'ln_final', 'text_projection', 'logit_scale', 'visual.ln_post',
                'visual.proj'
            ]
            mid_layers = [
                'visual.transformer.resblocks', 'transformer.resblocks'
            ]

            for name, param in mdl.clip.named_parameters():
                if any(name.find(n) == 0 for n in top_layers):
                    assert param.requires_grad is True
                elif any(name.find(n) == 0 for n in mid_layers):
                    layer_n = int(name.split('.resblocks.')[1].split('.')[0])
                    if layer_n >= frozen_layers:
                        assert param.requires_grad is True
                    else:
                        assert param.requires_grad is False
                else:
                    assert param.requires_grad is False
        else:
            assert all([p.requires_grad for p in mdl.clip.parameters()])

    check_frozen_layers(model, -1)

    model.frozen_layers = 0
    model.train()
    check_frozen_layers(model, 0)

    model.frozen_layers = 6
    model.train()
    check_frozen_layers(model, 6)

    model.frozen_layers = 12
    model.train()
    check_frozen_layers(model, 12)
