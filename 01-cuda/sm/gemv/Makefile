CUDA_PATH ?= /usr/local/cuda-12.8
NVCC = $(CUDA_PATH)/bin/nvcc
NVCC_FLAGS = -O3 -arch=native

# 目标文件
TARGETS = cublasLt verify_sm persistent_gemv

all: $(TARGETS)

cublasLt: cublasLt.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $< -lcublasLt

verify_sm: verify_sm.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $< -lcublasLt -lnvToolsExt

persistent_gemv: persistent_gemv.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $< -lnvToolsExt

run: cublasLt
	./cublasLt

run-verify: verify_sm
	./verify_sm

clean:
	rm -f $(TARGETS)

.PHONY: all clean run run-verify
