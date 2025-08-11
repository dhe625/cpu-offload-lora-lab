// custom_kernel/csrc/busy_wait.cu
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

__device__ __forceinline__ void device_pause(int cycles) {
  const unsigned long long start = clock64();
  while ((clock64() - start) < static_cast<unsigned long long>(cycles)) { }
}

__global__ void busy_wait_kernel(int* flag,
                                 int  backoff_cycles,
                                 unsigned long long timeout_cycles,
                                 int  auto_clear,
                                 int* status_out)
{
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    int st = 0;
    const unsigned long long start = clock64();

    while (atomicCAS(flag, 1, 1) != 1) {
      if (timeout_cycles) {
        unsigned long long now = clock64();
        if ((now - start) >= timeout_cycles) {
          st = 2;
          if (status_out) *status_out = st;
          return;
        }
      }
      if (backoff_cycles > 0) device_pause(backoff_cycles);
    }

    if (auto_clear) {
      atomicExch(flag, 0);
      __threadfence_system();
    }
    st = 1;
    if (status_out) *status_out = st;
  }
}

static void wait_func(torch::Tensor flag,
                      int backoff_cycles = 128,
                      unsigned long long timeout_cycles = 0ULL,
                      bool auto_clear = false,
                      torch::optional<torch::Tensor> status = torch::nullopt)
{
  TORCH_CHECK(flag.is_cuda(), "wait(flag): flag must be CUDA tensor");
  TORCH_CHECK(flag.scalar_type() == torch::kInt32 && flag.numel() == 1, "wait(flag): int32, numel=1");
  int* status_ptr = nullptr;
  if (status.has_value()) {
    TORCH_CHECK(status->is_cuda() && status->scalar_type() == torch::kInt32 && status->numel() == 1,
                "status must be CUDA int32 tensor of numel=1");
    status_ptr = status->data_ptr<int>();
    cudaMemsetAsync(status_ptr, 0, sizeof(int), at::cuda::getCurrentCUDAStream());
  }
  auto s = at::cuda::getCurrentCUDAStream();
  busy_wait_kernel<<<1, 1, 0, s>>>(flag.data_ptr<int>(),
                                   backoff_cycles,
                                   timeout_cycles,
                                   (int)auto_clear,
                                   status_ptr);
}

__global__ void set_flag_kernel(int* flag) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *flag = 1;
    __threadfence_system();
  }
}
static void set_flag_func(torch::Tensor flag) {
  TORCH_CHECK(flag.is_cuda() && flag.scalar_type()==torch::kInt32 && flag.numel()==1, "set_flag: bad flag");
  auto s = at::cuda::getCurrentCUDAStream();
  set_flag_kernel<<<1,1,0,s>>>(flag.data_ptr<int>());
}

__global__ void clear_flag_kernel(int* flag) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *flag = 0;
    __threadfence_system();
  }
}
static void clear_flag_func(torch::Tensor flag) {
  TORCH_CHECK(flag.is_cuda() && flag.scalar_type()==torch::kInt32 && flag.numel()==1, "clear_flag: bad flag");
  auto s = at::cuda::getCurrentCUDAStream();
  clear_flag_kernel<<<1,1,0,s>>>(flag.data_ptr<int>());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("wait",
        &wait_func,
        py::arg("flag"),
        py::arg("backoff_cycles")    = 128,
        py::arg("timeout_cycles")    = 0ULL,
        py::arg("auto_clear")        = false,
        py::arg("status")            = py::none(),
        "Busy-wait on device flag with optional timeout/auto_clear/status");
  m.def("set_flag",   &set_flag_func,   "Set flag=1");
  m.def("clear_flag", &clear_flag_func, "Set flag=0");
}