cmake_minimum_required(VERSION 3.18)
project(topk_benchmark CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES OFF)  # Use native arch
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -use_fast_math")

add_executable(topk_benchmark topk_benchmark.cu)

set_target_properties(topk_benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
