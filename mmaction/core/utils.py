# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn.functional as F


def stack_batch(tensors, size_divisor=0, pad_value=0):
    """Stack multiple tensors to form a batch and pad the images to the max
    shape use the right bottom padding mode in these images.

    Args:
        tensors (Tensor, List[Tensor]): The input multiple tensors.
            each is a CHW 3D-tensor.
        size_divisor: If `size_divisor > 0`, add padding to ensure
            the common height and width is divisible by `size_divisor`.
            This depends on the model and many models need a
            divisibility of 32. Default: 0
        pad_value: The padding value. Default: 0

    Returns:
       Tensor: The 4D-tensor.
    """
    assert isinstance(tensors, list)
    assert len(set([tensor.ndim for tensor in tensors])) == 1
    assert len(set([tensor.shape[:-2] for tensor in tensors])) == 1

    tensor_sizes = [(tensor.shape[-2], tensor.shape[-1]) for tensor in tensors]
    max_size = np.stack(tensor_sizes).max(0)

    if size_divisor > 1:
        stride = size_divisor
        # the last two dims are H,W, both subject to divisibility requirement
        max_size = (max_size + (stride - 1)) // stride * stride

    padded_samples = []
    for tensor in tensors:
        padding_size = [
            0, max_size[-1] - tensor.shape[-1], 0,
            max_size[-2] - tensor.shape[-2]
        ]
        if sum(padding_size) == 0:
            padded_samples.append(tensor)
        else:
            padded_samples.append(F.pad(tensor, padding_size, value=pad_value))

    return torch.stack(padded_samples, dim=0)
