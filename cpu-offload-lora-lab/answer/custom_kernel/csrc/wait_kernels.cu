#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void busy_wait(scalar_t* wait_flag) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    // printf("Before atomicCAS in busy_wait: flag = %d\n", (int)(*wait_flag));
    while (atomicAdd_system((int *)wait_flag, 0) == 0) 
    { 
      __nanosleep(100);
    }
    atomicAdd_system((int *)wait_flag, -1);
    // printf("After atomicCAS in busy_wait: flag = %d\n", (int)(*wait_flag));
  }
}

template <typename scalar_t>
__global__ void set(scalar_t* flag, int val) {
  // printf("Before atomicCAS in set: flag = %d\n", (int)(*flag));
  atomicCAS_system((int *)flag, *(int *)flag, val);
  // printf("After atomicCAS in set: flag = %d\n", (int)(*flag));
}


void wait_func(torch::Tensor& wait_flag) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_INTEGRAL_TYPES(
    wait_flag.scalar_type(),
    "busy_wait",
    [&] {
      busy_wait<scalar_t><<<1, 1, 0, stream>>>(
        wait_flag.data_ptr<scalar_t>());
    }
  );
}


void set_func(torch::Tensor& flag) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_INTEGRAL_TYPES(
    flag.scalar_type(), 
    "set",
    [&] {
      set<scalar_t><<<1, 1, 0, stream>>>(
        flag.data_ptr<scalar_t>(),
        1);
    }
  );
}