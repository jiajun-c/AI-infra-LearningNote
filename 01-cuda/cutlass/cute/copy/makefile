# 编译器设置
NVCC = nvcc

# 编译选项
# -std=c++17: CuTe 需要 C++17 或更高版本
# -O3: 开启最高级别的优化
# -lineinfo: (可选) 生成行号信息，方便使用 Nsight Compute 等工具进行性能分析
NVCC_FLAGS = -std=c++17 -O3 -lineinfo --expt-extended-lambda

# 架构设置 (重要！)
# 你的目标 GPU 架构。
# 例如：
# -gencode arch=compute_80,code=sm_80 (Ampere: A100)
# -gencode arch=compute_89,code=sm_89 (Ada Lovelace: RTX 4090, L40)
# -gencode arch=compute_90,code=sm_90 (Hopper: H100)
# 根据你的实际 GPU 修改。这里以 Ampere 和 Hopper 为例。
GENCODE_FLAGS = -gencode arch=compute_80,code=sm_80 \
                -gencode arch=compute_90,code=sm_90

# 包含路径 (Include Paths)
# **必须修改此处**：指向你本地 CUTLASS 仓库的 include 目录。
# 例如: CUTLASS_INC = -I/path/to/your/cutlass/include
CUTLASS_INC = -I/volume/code/chengjiajun/cutlass/include

# 目标文件列表
TARGETS = simple_copy tile_copy

# 默认构建目标
all: $(TARGETS)

# 通用编译规则：%.cu -> %
%: %.cu
	$(NVCC) $(NVCC_FLAGS) $(GENCODE_FLAGS) $(CUTLASS_INC) $< -o $@

# 运行规则
run_simple: simple_copy
	./simple_copy

run_tile: tile_copy
	./tile_copy

# 清理规则
clean:
	rm -f $(TARGETS)

.PHONY: all run_simple run_tile clean