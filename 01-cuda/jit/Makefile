# CUDA JIT编译 (NVRTC) 示例 Makefile
# 注意: 主机代码由nvcc AOT编译，JIT部分在运行时由NVRTC完成

NVCC      := nvcc
TARGET    := jit_demo
SRC       := nvrtc_jit_demo.cu

# -lnvrtc: NVRTC运行时编译库 (nvrtcCompileProgram等)
# -lcuda:  CUDA Driver API (cuModuleLoadData, cuLaunchKernel等)
NVCCFLAGS := -std=c++17 -O3
LDFLAGS   := -lnvrtc -lcuda

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LDFLAGS)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
