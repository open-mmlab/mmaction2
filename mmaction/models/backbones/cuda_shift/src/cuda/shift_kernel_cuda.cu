#include "shift_kernel_cuda.h"
#include <cstdio>
#include <cstring>

#define CUDA_KERNEL_LOOP(i, n)     \
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
                i < (n);                                    \
                i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;
inline int GET_BLOCKS(const int N)
{
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

__global__ void ShiftDataCudaForwardKernel(int n, float* data, int* shift, const int batch_size, const int channels, const int tsize, const int hwsize, const int groupsize, int groupchannel, float* out){
        CUDA_KERNEL_LOOP(index, n)
        {
        const int hw_index = index % hwsize;
        const int j = (index / hwsize) % channels;

        const int n_index = (index / hwsize / channels) % batch_size;
        int group_id = j / groupchannel;
        int t_shift = shift[n_index * groupsize + group_id];
        int offset = n_index * tsize * hwsize * channels + hwsize* j + hw_index;
        for(int i=0; i < tsize; i++)
                {
                int now_t = i + t_shift;
                int data_id =  i * hwsize * channels + offset;
                if (now_t < 0 || now_t >= tsize) {
                    continue;
                }
                int out_id =  now_t * hwsize * channels +offset;
                out[out_id] = data[data_id];
                }
        }
}


__global__ void ShiftDataCudaBackwardKernel(int n, float* data, int* shift, const int batch_size, const int channels, const int tsize, const int hwsize, const int groupsize, int groupchannel, float* out){
        CUDA_KERNEL_LOOP(index, n)
        {
        const int hw_index = index % hwsize;
        const int j = (index / hwsize) % channels;
        const int n_index = (index / hwsize / channels) % batch_size;
        int group_id = j / groupchannel;
        int t_shift = shift[n_index * groupsize + group_id];
        int offset = n_index * tsize * hwsize * channels + hwsize* j + hw_index;
        for(int i=0; i < tsize; i++)
                {
                int now_t = i - t_shift;
                int data_id =  i * hwsize * channels + offset;
                if (now_t < 0 || now_t >= tsize) {
                    continue;
                }
                int out_id =  now_t * hwsize * channels +offset;
                out[out_id] = data[data_id];

                }
        }
}

void ShiftDataCudaForward(cudaStream_t stream,
                          float* data,
                          int* shift,
                          const int batch_size,
                          const int channels,
                          const int tsize,
                          const int hwsize,
                          const int groupsize,
                          float* out){
        const int num_kernels = batch_size * hwsize * channels;
        int groupchannel = channels / groupsize;
        ShiftDataCudaForwardKernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(num_kernels, data, shift, batch_size, channels, tsize, hwsize, groupsize, groupchannel, out);
}

void ShiftDataCudaBackward(cudaStream_t stream,
                          float* data,
                          int* shift,
                          const int batch_size,
                          const int channels,
                          const int tsize,
                          const int hwsize,
                          const int groupsize,
                          float* out){
        const int num_kernels = batch_size * hwsize * channels;
        int groupchannel = channels / groupsize;
        ShiftDataCudaBackwardKernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(num_kernels, data, shift, batch_size, channels, tsize, hwsize, groupsize, groupchannel, out);
}
