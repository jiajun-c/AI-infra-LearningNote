# CUDA Driver API Stream Memory Operations — 编译 & 运行
#
# make all      编译全部
# make run      编译并运行全部
# make 01       编译例1: cuStreamWriteValue32 + cuStreamWaitValue32 (EQ)
# make 02       编译例2: 多阶段流水线 (GEQ)
# make 03       编译例3: cuStreamBatchMemOp 批量同步
# make clean

NVCC      ?= nvcc
ARCH      ?= native
NVCCFLAGS  = -arch=$(ARCH) -O3 -lcuda

TARGETS    = 01-write_wait 02-pipeline 03-batch

.PHONY: all run clean

all: $(TARGETS)

run: all
	@echo "=== 运行 01: WriteValue + WaitValue (EQ) ==="
	./01-write_wait
	@echo ""
	@echo "=== 运行 02: Pipeline (GEQ) ==="
	./02-pipeline
	@echo ""
	@echo "=== 运行 03: BatchMemOp ==="
	./03-batch

01-write_wait: 01-write_wait.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

02-pipeline: 02-pipeline.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

03-batch: 03-batch.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

01: 01-write_wait
	./01-write_wait

02: 02-pipeline
	./02-pipeline

03: 03-batch
	./03-batch

clean:
	rm -f $(TARGETS)
