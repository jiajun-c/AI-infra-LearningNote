# VLLM sleep mode

vLLM sleep mode 允许vllm暂时释放GPU显存给其他的任务，如RLHF训练

- 0: 仅暂停调度器，显存不变
- 1: 权重offload到CPU RAM，丢弃KVcache
- 2: 丢弃一切，不保存CPU副本

如下所示是一个完整的流程

```python
LLM.sleep(level)                                    # 用户 API
  ↓
LLMEngine/AsyncLLM                                  # 引擎层
  ↓
EngineCore.sleep()                                  # 核心协调
  ├─→ pause_scheduler()                             # 先停调度
  └─→ model_executor.sleep(level)                   # 再处理显存
          ↓
    Executor.collective_rpc("sleep")               # 分布式广播
          ↓
    GPUWorker.sleep(level)                         # Worker 执行
          ↓
    CuMemAllocator.sleep(offload_tags)             # 实际显存操作
```

## 1. 调度器暂停

vllm/vllm/v1/engine/core.py 

如下所示，定义了UNPAUSED，PAUSED_NEW, PAUSED_ALL 三种状态

- PAUSED_NEW: 只拒绝新请求，继续执行已有的请求
- PAUSED_ALL: 完全停止调度，中断当前的请求

```python
    def pause_scheduler(
        self, mode: PauseMode = "abort", clear_cache: bool = True
    ) -> Future | None:
        """Pause generation; behavior depends on mode.

        All pause modes queue new adds -- "abort" and "keep" skip step();
        "wait" allows step() so in-flight requests can drain.

        - ``abort``: Set PAUSED_NEW, abort all requests, wait for abort
          outputs to be sent (when running with output_queue), optionally
          clear caches, then complete the returned Future.
        - ``wait``: Set PAUSED_NEW (queue adds, keep stepping); when drained,
          optionally clear caches, then complete the returned Future.
        - ``keep``: Set PAUSED_ALL; return a Future that completes when the
          output queue is empty.
        """
        if mode not in ("keep", "abort", "wait"):
            raise ValueError(f"Invalid pause mode: {mode}")
        if mode == "wait":
            raise ValueError("'wait' mode can't be used in inproc-engine mode")

        if mode == "abort":
            self.scheduler.finish_requests(None, RequestStatus.FINISHED_ABORTED)

        pause_state = PauseState.PAUSED_ALL if mode == "keep" else PauseState.PAUSED_NEW
        self.scheduler.set_pause_state(pause_state)
        if clear_cache:
            self._reset_caches()

        return None
```

## 2. GPU 显存管理

cuMemAllocator 是一个单例，通过CUDA Virtual Memory Management API 管理所有显存。

每个AllocationData记录

- handle：CUDA虚拟内存句柄
- tag：标签（"weights" 或 "kvcache")
- cpu_backup_tensor: CPU备份 

sleep 流程

```python3
def sleep(self, offload_tags=("weights",)):
    for handle, data in self.allocations.items():
        if data.tag in offload_tags:
            # GPU → CPU: cudaMemcpy 拷贝到 pin_memory tensor
            data.cpu_backup_tensor = torch.empty(..., pin_memory=True)
            cudaMemcpy(cpu_ptr, gpu_ptr, size)
        # else: 直接丢弃，不保留副本
        
        unmap_and_release(handle)  # 释放虚拟地址映射
    
    torch.cuda.empty_cache()
```

wake_up 