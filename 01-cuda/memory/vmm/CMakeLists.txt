cmake_minimum_required(VERSION 3.18)
project(vmm_demo LANGUAGES C CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 80)  # Ampere; 改成你的 GPU sm 版本

add_executable(vmm_demo vmm_demo.cu)

target_compile_options(vmm_demo PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    -lineinfo
    --expt-relaxed-constexpr
>)

# VMM API 在 libcuda.so（驱动库），不是 libcudart
target_link_libraries(vmm_demo PRIVATE cuda)
