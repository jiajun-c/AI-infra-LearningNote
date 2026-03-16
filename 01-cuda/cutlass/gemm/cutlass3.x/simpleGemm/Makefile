# Makefile for CUTLASS 3.x Simple GEMM Demo (SM90 / Hopper)
#
# Usage:
#   make            # Build the demo
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
NVCCFLAGS   := -std=c++17 -O3 \
               -arch=$(GPU_ARCH) \
               --expt-relaxed-constexpr \
               -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1

# ---- Include paths ----
INCLUDES    := -I$(CUTLASS_PATH)/include \
               -I$(CUTLASS_PATH)/tools/util/include \
               -I$(CUTLASS_PATH)/examples/common

# ---- Libraries ----
LDFLAGS     := -lcuda -lcublas

# ---- Targets ----
TARGET      := demo
SRC         := demo.cu

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
