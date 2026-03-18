# Makefile for WGMMA (Warp Group Matrix Multiply Accumulate) Demo
#
# Usage:
#   make            # Build
#   make run        # Build and run (default 4096x4096x4096)
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
NVCCFLAGS   := -std=c++20 -O3 \
               -arch=$(GPU_ARCH) \
               --expt-relaxed-constexpr \
               --expt-extended-lambda

# ---- Include paths ----
INCLUDES    := -I$(CUTLASS_PATH)/include \
               -I$(CUTLASS_PATH)/tools/util/include

# ---- Libraries ----
LDFLAGS     := -lcuda -lcublas

# ---- Targets ----
TARGET      := wgmma_bench
SRC         := demo.cu

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
