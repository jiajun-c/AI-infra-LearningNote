# Makefile for CUDA SM device tests

# Compiler and flags
NVCC := nvcc
CC := gcc

# CUDA paths (adjust if needed)
CUDA_PATH ?= /usr/local/cuda
CUDA_INCLUDE := -I$(CUDA_PATH)/include
CUDA_LIB_PATH := -L$(CUDA_PATH)/lib64

# libsmctrl paths
LIBSMCTRL_PATH ?= /volume/code/jjcheng/libsmctrl
LIBSMCTRL_INCLUDE := -I$(LIBSMCTRL_PATH)
LIBSMCTRL_LIB_PATH := -L$(LIBSMCTRL_PATH)

# Compiler flags
NVCC_FLAGS := -O3 -arch=sm_90 -diag-suppress 177 $(CUDA_INCLUDE) $(LIBSMCTRL_INCLUDE)
CC_FLAGS := -O3

# Libraries
LIBS := -lcuda -lcudart -lsmctrl

# Targets
all: main

# Main target (using Driver API)
main: main.cu
	$(NVCC) $(NVCC_FLAGS) $(CUDA_INCLUDE) -o $@ $< $(CUDA_LIB_PATH) $(LIBS)

# Clean
clean:
	rm -f main *.o

# Run
run: main
	./main

.PHONY: all clean run
