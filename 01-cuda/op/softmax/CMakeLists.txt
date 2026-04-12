cmake_minimum_required(VERSION 3.18)
project(online_softmax_test CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES OFF)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -use_fast_math")

add_executable(online_softmax_test online_softmax_test.cu)

set_target_properties(online_softmax_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
