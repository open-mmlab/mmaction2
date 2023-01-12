# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from tempfile import TemporaryDirectory
from unittest import TestCase

import torch
from parameterized import parameterized

from mmaction.apis import ActionRecogInferencer
from mmaction.structures import ActionDataSample
from mmaction.utils import register_all_modules


class TestInferencer(TestCase):

    def setUp(self):
        register_all_modules()

    @parameterized.expand([
        (('tsn'), ('tools/data/kinetics/label_map_k400.txt'), ('cpu', 'cuda'))
    ])
    def test_init_recognizer(self, config, lable_file, devices):

        for device in devices:
            if device == 'cuda' and not torch.cuda.is_available():
                # Skip the test if cuda is required but unavailable
                continue

            _ = ActionRecogInferencer(config, label=lable_file, device=device)

            # test `init_recognizer` with invalid config
            with self.assertRaisesRegex(ValueError, 'Cannot find model'):
                _ = ActionRecogInferencer(
                    'slowfast_config', label=lable_file, device=device)

    @parameterized.expand([
        (('tsn'), ('tools/data/kinetics/label_map_k400.txt'),
         ('demo/demo.mp4'), ('cpu', 'cuda'))
    ])
    def test_inference_recognizer(self, config, label_file, video_path,
                                  devices):

        with TemporaryDirectory() as tmp_dir:
            for device in devices:
                if device == 'cuda' and not torch.cuda.is_available():
                    # Skip the test if cuda is required but unavailable
                    continue

                # test video file input and return datasample
                inferencer = ActionRecogInferencer(
                    config, label=label_file, device=device)
                results = inferencer(
                    video_path, vid_out_dir=tmp_dir, return_datasamples=True)
                self.assertIn('predictions', results)
                self.assertIn('visualization', results)
                self.assertIsInstance(results['predictions'][0],
                                      ActionDataSample)
                assert osp.exists(osp.join(tmp_dir, osp.basename(video_path)))

                results = inferencer(
                    video_path, vid_out_dir=tmp_dir, out_type='gif')
                self.assertIsInstance(results['predictions'][0], dict)
                assert osp.exists(
                    osp.join(tmp_dir,
                             osp.basename(video_path).replace('mp4', 'gif')))

                # test np.ndarray input
                inferencer = ActionRecogInferencer(
                    config,
                    label=label_file,
                    device=device,
                    input_format='array')
                import decord
                import numpy as np
                video = decord.VideoReader(video_path)
                frames = [x.asnumpy()[..., ::-1] for x in video]
                frames = np.stack(frames)
                inferencer(frames, vid_out_dir=tmp_dir)
                assert osp.exists(osp.join(tmp_dir, '00000000.mp4'))
