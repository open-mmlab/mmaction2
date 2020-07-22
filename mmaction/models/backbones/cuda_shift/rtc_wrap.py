# Code for "Temporal Interlacing Network"
# Hao Shao, Shengju Qian, Yu Liu
# shaoh19@mails.tsinghua.edu.cn, sjqian@cse.cuhk.edu.hk, yuliu@ee.cuhk.edu.hk

import cudashift as backend
import torch
from torch.autograd import Function


class ShiftFeatureFunc(Function):

    def __init__(self):
        super(ShiftFeatureFunc, self).__init__()

    @staticmethod
    def forward(self, data, shift):
        if not data.is_cuda or not shift.is_cuda:
            raise NotImplementedError

        if data.requires_grad:
            self.save_for_backward(shift)

        out = torch.zeros_like(data)
        backend.shift_featuremap_cuda_forward(data, shift, out)
        return out

    @staticmethod
    def backward(self, grad_output):
        if not grad_output.is_cuda:
            raise NotImplementedError
        shift = self.saved_tensors[0]
        data_grad_input = grad_output.new(*grad_output.size()).zero_()
        shift_grad_input = shift.new(*shift.size()).zero_()
        backend.shift_featuremap_cuda_backward(grad_output, shift,
                                               data_grad_input)
        return data_grad_input, shift_grad_input
