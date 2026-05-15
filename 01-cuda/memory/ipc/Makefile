NVCC ?= nvcc
CXXFLAGS ?= -O2 -std=c++17

TARGET := cuda_ipc_p2p_demo
SRC := cuda_ipc_p2p_demo.cu

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CXXFLAGS) $< -o $@

clean:
	rm -f $(TARGET)
