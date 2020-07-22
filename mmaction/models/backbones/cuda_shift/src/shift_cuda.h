at::Tensor shift_featuremap_cuda_forward(at::Tensor &data,
                                   at::Tensor &shift, at::Tensor &out);

at::Tensor shift_featuremap_cuda_backward(at::Tensor &grad_output,
                                   at::Tensor &shift, at::Tensor &grad_input);
