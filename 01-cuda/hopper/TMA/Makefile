# Makefile for TMA (Tensor Memory Accelerator) Hello World Demo
#
# Usage:
#   make            # Build
#   make run        # Build and run
#   make clean      # Remove build artifacts
#
# Prerequisites:
#   - CUDA 12.0+
#   - NVIDIA Hopper GPU (compute capability 9.0)
#   - CUTLASS source tree (set CUTLASS_PATH if not at default location)

# ---- Toolchain ----
NVCC        := nvcc

# ---- CUTLASS paths ----
CUTLASS_PATH ?= /volume/code/chengjiajun/workspace/cutlass

# ---- Architecture (Hopper SM90) ----
GPU_ARCH    := sm_90a

# ---- Compiler flags ----
# 注意：main.cu 使用了 auto 参数的 __global__ 函数，需要 C++20 支持
NVCCFLAGS   := -std=c++20 -O3 \
               -arch=$(GPU_ARCH) \
               --expt-relaxed-constexpr \
               --expt-extended-lambda

# ---- Include paths ----
INCLUDES    := -I$(CUTLASS_PATH)/include \
               -I$(CUTLASS_PATH)/tools/util/include

# ---- Libraries ----
LDFLAGS     := -lcuda

# ---- Targets ----
TARGET      := tma
SRC         := main.cu

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
