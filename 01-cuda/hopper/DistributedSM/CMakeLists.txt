cmake_minimum_required(VERSION 3.22)

project(tma CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES 90a)

# find_package(cuBLAS REQUIRED)
# include_directories(/home/star/workspace/cuda/cuda_blas/include)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")

add_executable(tma main.cu)
target_link_libraries(tma ${CUBLAS_LIBRARIES})