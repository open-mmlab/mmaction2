# Copyright (c) OpenMMLab. All rights reserved.
from .inference import inference_recognizer, init_recognizer
from .test import multi_gpu_test, single_gpu_test
from .train import init_random_seed, train_model

__all__ = [
    'train_model', 'init_recognizer', 'inference_recognizer', 'multi_gpu_test',
    'single_gpu_test', 'init_random_seed'
]
