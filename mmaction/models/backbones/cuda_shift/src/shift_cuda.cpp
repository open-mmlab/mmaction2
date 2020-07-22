#include <THC/THC.h>
#include <torch/extension.h>
#include "cuda/shift_kernel_cuda.h"
#include <ATen/cuda/CUDAContext.h>

extern THCState *state;
at::Tensor shift_featuremap_cuda_forward(const at::Tensor &data, const at::Tensor &shift, const at::Tensor &out)
{
    THArgCheck(data.is_contiguous(), 1, "data tensor has to be contiguous");
    THArgCheck(shift.is_contiguous(), 1, "shift tensor has to be contiguous");

    int batch_size = data.size(0);
    int channels = data.size(2);
    int tsize = data.size(1);
    int hwsize = data.size(3);
    int groupsize = shift.size(1);

    ShiftDataCudaForward(THCState_getCurrentStream(state),
                        data.data<float>(),
                        shift.data<int>(),
                        batch_size,
                        channels,
                        tsize,
                        hwsize,
                        groupsize,
                        out.data<float>());
    return out;
}

at::Tensor shift_featuremap_cuda_backward(const at::Tensor &grad_output, const at::Tensor &shift, const at::Tensor &grad_input)
{
    THArgCheck(grad_output.is_contiguous(), 1, "data tensor has to be contiguous");
    THArgCheck(shift.is_contiguous(), 1, "shift tensor has to be contiguous");

    int batch_size = grad_output.size(0);
    int channels = grad_output.size(2);
    int tsize = grad_output.size(1);
    int hwsize = grad_output.size(3);
    int groupsize = shift.size(1);

    ShiftDataCudaBackward(THCState_getCurrentStream(state),
                        grad_output.data<float>(),
                        shift.data<int>(),
                        batch_size,
                        channels,
                        tsize,
                        hwsize,
                        groupsize,
                        grad_input.data<float>());
    return grad_input;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("shift_featuremap_cuda_forward", &shift_featuremap_cuda_forward, "shift_featuremap_cuda_forward");
  m.def("shift_featuremap_cuda_backward", &shift_featuremap_cuda_backward, "shift_featuremap_cuda_backward");
}
