# Makefile for CUTLASS 2.x Device GEMM benchmarks
#
# Usage:
#   make            # Build all targets
#   make sgemm      # Build sgemm only
#   make run        # Build and run sgemm
#   make clean      # Remove build artifacts

# ---- Toolchain ----
NVCC        := nvcc

# ---- CUTLASS paths ----
CUTLASS_PATH ?= /volume/code/chengjiajun/workspace/cutlass

# ---- Architecture (adjust for your GPU) ----
GPU_ARCH    := sm_80

# ---- Compiler flags ----
NVCCFLAGS   := -std=c++17 -O3 \
               -arch=$(GPU_ARCH) \
               --expt-relaxed-constexpr

# ---- Include paths ----
INCLUDES    := -I$(CUTLASS_PATH)/include \
               -I$(CUTLASS_PATH)/tools/util/include

# ---- Libraries ----
LDFLAGS     := -lcublas

# ---- Targets ----
TARGETS     := sgemm

.PHONY: all run clean

all: $(TARGETS)

sgemm: sgemm.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS)

run: sgemm
	./sgemm

clean:
	rm -f $(TARGETS)
