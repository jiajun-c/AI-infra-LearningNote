# Makefile for CUTLASS GEMM benchmark (CUTLASS vs cuBLAS)
#
# Usage:
#   make            # Build the benchmark
#   make run        # Build and run with default parameters
#   make clean      # Remove build artifacts
#
# Custom run:
#   ./pingpong --m=4096 --n=4096 --k=4096 --iterations=50

# ---- Toolchain ----
NVCC        := nvcc
CUDA_PATH   ?= $(shell dirname $(shell dirname $(shell which nvcc)))

# ---- CUTLASS paths ----
CUTLASS_PATH ?= /volume/code/chengjiajun/workspace/cutlass

# ---- Architecture ----
GPU_ARCH    := sm_90a

# ---- Compiler flags ----
NVCCFLAGS   := -std=c++17 -O3 \
               -arch=$(GPU_ARCH) \
               --expt-relaxed-constexpr \
               -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1

# ---- Include paths ----
INCLUDES    := -I$(CUTLASS_PATH)/include \
               -I$(CUTLASS_PATH)/tools/util/include

# ---- Libraries ----
LDFLAGS     := -lcublas -lcuda

# ---- Targets ----
TARGET      := pingpong
SRC         := pingpong.cu

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS) 2>&1 | tee out

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) out
