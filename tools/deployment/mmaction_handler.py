# Copyright (c) OpenMMLab. All rights reserved.
import base64
import os
import os.path as osp
import warnings

import decord
import numpy as np
import torch

from mmaction.apis import inference_recognizer, init_recognizer  # noqa: F401

try:
    from ts.torch_handler.base_handler import BaseHandler
except ImportError:
    raise ImportError('`ts` is required. Try: pip install ts.')


class MMActionHandler(BaseHandler):

    def initialize(self, context):
        properties = context.system_properties
        self.map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.map_location + ':' +
                                   str(properties.get('gpu_id')) if torch.cuda.
                                   is_available() else self.map_location)
        self.manifest = context.manifest

        model_dir = properties.get('model_dir')
        serialized_file = self.manifest['model']['serializedFile']
        checkpoint = os.path.join(model_dir, serialized_file)
        self.config_file = os.path.join(model_dir, 'config.py')

        mapping_file_path = osp.join(model_dir, 'label_map.txt')
        if not os.path.isfile(mapping_file_path):
            warnings.warn('Missing the label_map.txt file. '
                          'Inference output will not include class name.')
            self.mapping = None
        else:
            lines = open(mapping_file_path).readlines()
            self.mapping = [x.strip() for x in lines]

        self.model = init_recognizer(self.config_file, checkpoint, self.device)
        self.initialized = True

    def preprocess(self, data):
        videos = []

        for row in data:
            video = row.get('data') or row.get('body')
            if isinstance(video, str):
                video = base64.b64decode(video)
            # First save the bytes as a tmp file
            with open('/tmp/tmp.mp4', 'wb') as fout:
                fout.write(video)

            video = decord.VideoReader('/tmp/tmp.mp4')
            frames = [x.asnumpy() for x in video]
            videos.append(np.stack(frames))

        return videos

    def inference(self, data, *args, **kwargs):
        results = [inference_recognizer(self.model, item) for item in data]
        return results

    def postprocess(self, data):
        # Format output following the example ObjectDetectionHandler format
        output = []
        for video_idx, video_result in enumerate(data):
            output.append([])
            assert isinstance(video_result, list)

            output[video_idx] = {
                self.mapping[x[0]] if self.mapping else x[0]: float(x[1])
                for x in video_result
            }

        return output
