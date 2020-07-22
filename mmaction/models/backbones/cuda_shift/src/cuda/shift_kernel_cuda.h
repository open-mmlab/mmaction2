#ifndef Shift_FeatureMap_CUDA
#define Shift_FeatureMap_CUDA

#ifdef __cplusplus
extern "C"
{
#endif

void ShiftDataCudaForward(cudaStream_t stream,
                          float* data,
                          int* shift,
                          const int batch_size,
                          const int channels,
                          const int tsize,
                          const int hwsize,
                          const int groupsize,
                          float* out);

void ShiftDataCudaBackward(cudaStream_t stream,
                          float* grad_output,
                          int* shift,
                          const int batch_size,
                          const int channels,
                          const int tsize,
                          const int hwsize,
                          const int groupsize,
                          float* grad_input);
#ifdef __cplusplus
}
#endif

#endif
