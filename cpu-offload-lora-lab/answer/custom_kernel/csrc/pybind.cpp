#include <torch/extension.h>

void wait_func(torch::Tensor& wait_flag);
void set_func(torch::Tensor& flag);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  pybind11::module cuda_utils = m.def_submodule("cuda_utils", "PyTorch CUDA utility functions or objects");
  cuda_utils.def(
    "wait", 
    &wait_func, 
    "Wait for flag to be set");
  cuda_utils.def(
    "set", 
    &set_func, 
    "Set for flag");
}