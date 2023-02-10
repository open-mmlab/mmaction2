# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from tempfile import TemporaryDirectory
from unittest import TestCase

import torch
from parameterized import parameterized

from mmaction.apis import MMAction2Inferencer


class TestMMActionInferencer(TestCase):

    def test_init_recognizer(self):
        # Initialzied by alias
        _ = MMAction2Inferencer(rec='tsn')

        # Initialzied by config
        _ = MMAction2Inferencer(
            rec='tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb'
        )  # noqa: E501

        with self.assertRaisesRegex(ValueError,
                                    'rec algorithm should provided.'):
            _ = MMAction2Inferencer()

    @parameterized.expand([
        (('tsn'), ('tools/data/kinetics/label_map_k400.txt'),
         ('demo/demo.mp4'), ('cpu', 'cuda'))
    ])
    def test_infer_recognizer(self, config, label_file, video_path, devices):
        with TemporaryDirectory() as tmp_dir:
            for device in devices:
                if device == 'cuda' and not torch.cuda.is_available():
                    # Skip the test if cuda is required but unavailable
                    continue

                # test video file input and return datasample
                inferencer = MMAction2Inferencer(
                    config, label_file=label_file, device=device)
                results = inferencer(video_path, vid_out_dir=tmp_dir)
                self.assertIn('predictions', results)
                self.assertIn('visualization', results)
                assert osp.exists(osp.join(tmp_dir, osp.basename(video_path)))

                results = inferencer(
                    video_path, vid_out_dir=tmp_dir, out_type='gif')
                self.assertIsInstance(results['predictions'][0], dict)
                assert osp.exists(
                    osp.join(tmp_dir,
                             osp.basename(video_path).replace('mp4', 'gif')))

                # test np.ndarray input
                inferencer = MMAction2Inferencer(
                    config,
                    label_file=label_file,
                    device=device,
                    input_format='array')
                import decord
                import numpy as np
                video = decord.VideoReader(video_path)
                frames = [x.asnumpy()[..., ::-1] for x in video]
                frames = np.stack(frames)
                inferencer(frames, vid_out_dir=tmp_dir)
                assert osp.exists(osp.join(tmp_dir, '00000000.mp4'))
