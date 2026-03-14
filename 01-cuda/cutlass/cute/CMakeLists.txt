cmake_minimum_required(VERSION 3.18)
project(MyCuteProject LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 90) # 建议先改回 80 试试，除非你确定有 H100

# 👇 新增这行：强制 nvcc 使用 g++-11 作为宿主编译器
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")

# ... (后面保持不变)
# 2. 定义你的 CUTLASS 路径变量
# 假设你的目录结构是 .../cutlass/include/cute/...
set(CUTLASS_HOME "/staff/chengjiajun/local")

# 3. 创建接口库并添加 include 路径
add_library(cutlass INTERFACE)
target_include_directories(cutlass INTERFACE "${CUTLASS_HOME}/include")

# 4. 链接
add_executable(main main.cu)
target_link_libraries(main PRIVATE cutlass)