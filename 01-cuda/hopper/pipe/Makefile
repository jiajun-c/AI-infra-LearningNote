# Makefile for Double-Buffered Pipeline Demo
#
# Usage:
#   make            # Build
#   make run        # Build and run (default 1M elements)
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

# ---- Targets ----
TARGET      := pipe_bench
SRC         := pipe.cu

TMA_TARGET  := pipe_tma_bench
TMA_SRC     := pipe_tma.cu

.PHONY: all run pipe_tma run_tma clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS)

pipe_tma: $(TMA_TARGET)

$(TMA_TARGET): $(TMA_SRC)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS)

run: $(TARGET)
	./$(TARGET)

run_tma: $(TMA_TARGET)
	./$(TMA_TARGET)

clean:
	rm -f $(TARGET) $(TMA_TARGET)
