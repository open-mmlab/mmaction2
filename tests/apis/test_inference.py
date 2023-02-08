# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from pathlib import Path
from unittest import TestCase

import torch
from parameterized import parameterized

from mmaction.apis import inference_recognizer, init_recognizer
from mmaction.structures import ActionDataSample


class TestInference(TestCase):

    @parameterized.expand([(('configs/recognition/tsn/'
                             'tsn_imagenet-pretrained-r50_8xb32-'
                             '1x1x3-100e_kinetics400-rgb.py'), ('cpu', 'cuda'))
                           ])
    def test_init_recognizer(self, config, devices):
        project_dir = osp.abspath(osp.dirname(osp.dirname(__file__)))
        project_dir = osp.join(project_dir, '..')
        config_file = osp.join(project_dir, config)

        for device in devices:
            if device == 'cuda' and not torch.cuda.is_available():
                # Skip the test if cuda is required but unavailable
                continue

            # test `init_recognizer` with str path
            _ = init_recognizer(config_file, device=device)

            # test `init_recognizer` with :obj:`Path`
            _ = init_recognizer(Path(config_file), device=device)

            # test `init_recognizer` with undesirable type
            with self.assertRaisesRegex(
                    TypeError, 'config must be a filename or Config object'):
                config_list = [config_file]
                _ = init_recognizer(config_list)

    @parameterized.expand([(('configs/recognition/tsn/'
                             'tsn_imagenet-pretrained-r50_8xb32-'
                             '1x1x3-100e_kinetics400-rgb.py'), 'demo/demo.mp4',
                            ('cpu', 'cuda'))])
    def test_inference_recognizer(self, config, video_path, devices):
        project_dir = osp.abspath(osp.dirname(osp.dirname(__file__)))
        project_dir = osp.join(project_dir, '..')
        config_file = osp.join(project_dir, config)
        video_path = osp.join(project_dir, video_path)

        for device in devices:
            if device == 'cuda' and not torch.cuda.is_available():
                # Skip the test if cuda is required but unavailable
                continue
            model = init_recognizer(config_file, device=device)

            for ops in model.cfg.test_pipeline:
                if ops['type'] in ('TenCrop', 'ThreeCrop'):
                    # Use CenterCrop to reduce memory in order to pass CI
                    ops['type'] = 'CenterCrop'

            result = inference_recognizer(model, video_path)

            self.assertIsInstance(result, ActionDataSample)
            self.assertTrue(result.pred_scores.item.shape, (400, ))
