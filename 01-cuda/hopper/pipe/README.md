# hopper pipeline

## 1. PipelineState

这一段应该同时属于hopper和cutlass，在cutlass中被放入到`sm90_pipeline.hpp`中

`sm90_pipeline.hpp`中对`PipelineState`的定义如下所示

```cpp
// Circular Buffer Index + Associated Phase
// Assumes only one operation possible - i.e., ++
template<uint32_t Stages_>
struct PipelineState {

  static constexpr uint32_t Stages = Stages_;

  CUTLASS_DEVICE
  void operator++() {
    if constexpr (Stages > 0) {
      ++index_;
      ++count_;
      if (index_ == Stages) {
        index_ = 0;
        phase_ ^= 1;
      }
    }
  }

  CUTLASS_DEVICE
  PipelineState& operator+=(uint32_t num_iterations) {
    return advance(num_iterations);
  }

  CUTLASS_DEVICE
  PipelineState& operator=(PipelineState const& other) {
    index_ = other.index();
    phase_ = other.phase();
    count_ = other.count();
    return *this;
  }

  CUTLASS_DEVICE
  PipelineState& advance(uint32_t num_iterations) {
    if constexpr (Stages > 0) {
      // Number of iterations cross over the stage boundary => flipped phase
      if ((num_iterations < Stages) && (index_ + num_iterations) >= Stages ) {
        phase_ ^= 1;
      }
      // How many times number of iterations cross over the stage boundary and
      // end up on a odd number => flipped phase
      if ((num_iterations >= Stages) && (((index_ + num_iterations) / Stages) % 2) == 1) {
        phase_ ^= 1;
      }
      index_ = (index_ + num_iterations) % Stages;
      count_ += num_iterations;
    }
    return *this;
  }

  CUTLASS_DEVICE
  static PipelineState make_pipeline_state(PipelineState start_state, uint32_t num_iterations) {
    return start_state.advance(num_iterations);
  }
}
```

类成员大致有三个组成
- index_ : 当前处于流水线的第几个阶段（取值范围：$0$ 到 $Stages - 1$）。
- phase_ : 当前的相位/奇偶校验位（取值范围：$0$ 或 $1$），根据这个知道当前的生产者和消费者是否处于一个阶段
- count_：从开始到现在总共走过的迭代次数。

状态处理函数
- operator++() ：重载了自增的运算符，同时通过phase_来表示是在双缓冲的哪个阶段
- advance： 让状态机一次前进多步
- make_pipeline_state：让消费者根据初始化状态来快速生成一个延迟了N步的未来状
- index：获取到当前要写入的stage索引

## 2. ClusterBarrier

ClusterBarrier是Hopper中引入的集群级别的arrive-wait屏障
```cpp
struct ClusterBarrier {

  using ValueType = uint64_t;

protected:
  // Can never be initialized - can only be aliased to smem
  ValueType barrier_;

public:

  CUTLASS_DEVICE
  ClusterBarrier() = delete;

  CUTLASS_DEVICE
  void init(uint32_t arrive_count) const {
    ClusterBarrier::init(&this->barrier_, arrive_count);
  }

  CUTLASS_DEVICE
  bool test_wait(uint32_t phase, uint32_t pred=true) const {
    return ClusterBarrier::test_wait(&this->barrier_, phase, pred);
  }

  CUTLASS_DEVICE
  bool try_wait(uint32_t phase) const {
    return ClusterBarrier::try_wait(&this->barrier_, phase);
  }

  CUTLASS_DEVICE
  void wait(uint32_t phase) const {
    ClusterBarrier::wait(&this->barrier_, phase);
  }

  // Barrier arrive on local smem
  CUTLASS_DEVICE
  void arrive() const {
    ClusterBarrier::arrive(&this->barrier_);
  }

  // Remote SMEM arrive with a perdicate (usually done to pick the thread doing the arrive)
  CUTLASS_DEVICE
  void arrive(uint32_t cta_id, uint32_t pred = true ) const {
    ClusterBarrier::arrive(&this->barrier_, cta_id, pred);
  }

  //
  //  Static Versions
  //
  CUTLASS_HOST_DEVICE
  static void init(ValueType const* smem_ptr, uint32_t arrive_count) {
    CUTLASS_ASSERT(arrive_count != 0 && "Arrive count must be non-zero");
#if CUDA_BARRIER_ENABLED
    uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
    asm volatile(
        "{\n\t"
        "mbarrier.init.shared::cta.b64 [%1], %0; \n"
        "}"
        :
        : "r"(arrive_count), "r"(smem_addr)
        : "memory");
    cutlass::arch::synclog_emit_cluster_barrier_init(__LINE__, smem_addr, arrive_count);
#else
    CUTLASS_NOT_IMPLEMENTED();
#endif
  }

  // Static version of wait - in case we don't want to burn a register
  CUTLASS_HOST_DEVICE
  static void wait(ValueType const* smem_ptr, uint32_t phase) {
#if CUDA_BARRIER_ENABLED
    uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
    cutlass::arch::synclog_emit_cluster_barrier_wait(__LINE__, smem_addr, phase);
    // Arbitrarily large timer value after which try-wait expires and re-tries.
    uint32_t ticks = 0x989680;
    asm volatile(
        "{\n\t"
        ".reg .pred       P1; \n\t"
        "LAB_WAIT: \n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2; \n\t"
        "@P1 bra DONE; \n\t"
        "bra     LAB_WAIT; \n\t"
        "DONE: \n\t"
        "}"
        :
        : "r"(smem_addr), "r"(phase), "r"(ticks)
        : "memory");

#else
    CUTLASS_NOT_IMPLEMENTED();
#endif
  }

  CUTLASS_HOST_DEVICE
  static bool test_wait(ValueType const* smem_ptr, uint32_t phase, uint32_t pred) {
#if CUDA_BARRIER_ENABLED
    uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
    cutlass::arch::synclog_emit_cluster_barrier_test_wait(__LINE__, smem_addr, phase, pred);
    uint32_t waitComplete;

    asm volatile(
        "{\n\t"
        ".reg .pred P1; \n\t"
        ".reg .pred P2; \n\t"
        "setp.eq.u32 P2, %3, 1;\n\t"
        "@P2 mbarrier.test_wait.parity.shared::cta.b64 P1, [%1], %2; \n\t"
        "selp.b32 %0, 1, 0, P1; \n\t"
        "}"
        : "=r"(waitComplete)
        : "r"(smem_addr), "r"(phase), "r"(pred)
        : "memory");

    return static_cast<bool>(waitComplete);
#else
    CUTLASS_NOT_IMPLEMENTED();
#endif
    return 0;
  }

  CUTLASS_HOST_DEVICE
  static bool try_wait(ValueType const* smem_ptr, uint32_t phase) {
#if CUDA_BARRIER_ENABLED
    uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
    cutlass::arch::synclog_emit_cluster_barrier_try_wait(__LINE__, smem_addr, phase);
    uint32_t waitComplete;

    asm volatile(
        "{\n\t"
        ".reg .pred P1; \n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2; \n\t"
        "selp.b32 %0, 1, 0, P1; \n\t"
        "}"
        : "=r"(waitComplete)
        : "r"(smem_addr), "r"(phase)
        : "memory");

    return static_cast<bool>(waitComplete);
#else
    CUTLASS_NOT_IMPLEMENTED();
#endif
    return 0;
  }

  // Static Predicated version of the above - in case we know the address.
  CUTLASS_HOST_DEVICE
  static void arrive(ValueType const* smem_ptr, uint32_t cta_id, uint32_t pred) {
#if CUDA_BARRIER_ENABLED
    uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
    if (pred) {
      asm volatile(
          "{\n\t"
          ".reg .b32 remAddr32;\n\t"
          "mapa.shared::cluster.u32  remAddr32, %0, %1;\n\t"
          "mbarrier.arrive.shared::cluster.b64  _, [remAddr32];\n\t"
          "}"
          :
          : "r"(smem_addr), "r"(cta_id)
          : "memory");
    }

    cutlass::arch::synclog_emit_cluster_barrier_arrive_cluster(__LINE__, smem_addr, cta_id, pred);
#else
    CUTLASS_NOT_IMPLEMENTED();
#endif
  }

  // Barrier arrive on local smem
  CUTLASS_HOST_DEVICE
  static void arrive(ValueType const* smem_ptr) {
#if CUDA_BARRIER_ENABLED
    uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
    asm volatile(
        "{\n\t"
        "mbarrier.arrive.shared::cta.b64 _, [%0];\n\t"
        "}"
        :
        : "r"(smem_addr)
        : "memory");
    cutlass::arch::synclog_emit_cluster_barrier_arrive(__LINE__, smem_addr);
#else
    CUTLASS_NOT_IMPLEMENTED();
#endif
  }

  CUTLASS_HOST_DEVICE
  static void invalidate(ValueType const* smem_ptr) {
#if CUDA_BARRIER_ENABLED
    uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
    asm volatile(
        "{\n\t"
        "mbarrier.inval.shared::cta.b64 [%0]; \n\t"
        "}"
        :
        : "r"(smem_addr)
        : "memory");
#else
    CUTLASS_NOT_IMPLEMENTED();
#endif
  }
};
```

- init: 初始化屏障，设置多少个线程到达之后屏障才会翻转
- wait：阻塞等待屏障进入指定
- arrive：线程在本CTA的共享内存上执行到达的操作
- arrive(cta_id, pred): 在远程共享内存上执行到达的操作

## 3. ClusterTransactionBarrier

继承自上面的ClusterBarrier，增加了到达+期望字节计数，其主要应用场景是到达计数


## 4. 双缓冲demo

使用一个warp进行数据的搬运（生产者），还有三个warp负责进行计算（消费者）

先定义mbarrier，下面的结构体需要被放入到共享内存中，因为 mbarrier的函数都带来shared的标签

```cpp
struct SharedStorage {
    alignas(128) float buf[PIPE_STAGES][TILE_SIZE];  // 双缓冲数据区，128 字节对齐以满足 cp.async 要求
    uint64_t producer_mbar[PIPE_STAGES];             // Producer barrier：通知 Consumer "数据已就绪"
    uint64_t consumer_mbar[PIPE_STAGES];             // Consumer barrier：通知 Producer "buffer 已释放，可覆写"
};
```

初始化mbarrier

```cpp
    if (tid == 0) {
        CUTE_UNROLL
        for (int pipe = 0; pipe < PIPE_STAGES; ++pipe) {
            ProducerBarType::init(&producer_mbar[pipe], PRODUCER_THREADS);  // 32
            ConsumerBarType::init(&consumer_mbar[pipe], CONSUMER_THREADS);  // 96
        }
    }
```

进行prefech，对于每个流水的stage标记一下arrive `ProducerBarType::arrive(&producer_mbar[pipe]);`

```cpp
        // ---- Prefetch 阶段：预填充所有流水线 stage ----
        //   在 Consumer 开始消费之前，将前 PIPE_STAGES 个 tile 加载到 SMEM
        CUTE_UNROLL
        for (int pipe = 0; pipe < PIPE_STAGES; ++pipe) {
            if (k_tile_count > 0) {
                int global_offset = load_tile * TILE_SIZE;

                // 32 个线程协作加载 128 个 float，每线程 4 个
                CUTE_UNROLL
                for (int i = 0; i < TILE_SIZE / PRODUCER_THREADS; ++i) {
                    int elem_idx = producer_lane + i * PRODUCER_THREADS;
                    if (global_offset + elem_idx < N) {
                        smem.buf[pipe][elem_idx] = in[global_offset + elem_idx];
                    } else {
                        smem.buf[pipe][elem_idx] = 0.0f;
                    }
                }

                // 所有 32 个 Producer 线程 arrive，通知 Consumer "数据已就绪"
                ProducerBarType::arrive(&producer_mbar[pipe]);
                --k_tile_count;
                load_tile += gridDim.x;
            }
        }
```

在生产者的主循环中，其会先获取到当前是在写那个阶段的pipe，然后去wait消费者保证一下当前的pipe已经被消费者消费过了，可以继续写入数据。完成当前阶段的数据写入后标记一下arrive，对write_state进行自增

```cpp
            ProducerBarType::arrive(&producer_mbar[write_pipe]);
            ++write_state;
```

如下所示是完整的生产者的代码

```cpp
        // ---- 主循环：等待 Consumer 释放 buffer → 加载新 tile → 通知 Consumer ----
        CUTE_NO_UNROLL
        while (k_tile_count > 0) {
            int write_pipe = write_state.index();

            // 等待 Consumer 完成对当前 stage 的消费（consumer_mbar 的 96 个线程都已 arrive）
            ConsumerBarType::wait(&consumer_mbar[write_pipe], write_state.phase());

            // 32 个线程协作将新 tile 从 GMEM 搬运到 SMEM
            int global_offset = load_tile * TILE_SIZE;
            CUTE_UNROLL
            for (int i = 0; i < TILE_SIZE / PRODUCER_THREADS; ++i) {
                int elem_idx = producer_lane + i * PRODUCER_THREADS;
                if (global_offset + elem_idx < N) {
                    smem.buf[write_pipe][elem_idx] = in[global_offset + elem_idx];
                } else {
                    smem.buf[write_pipe][elem_idx] = 0.0f;
                }
            }

            // 通知 Consumer "新数据已就绪"
            ProducerBarType::arrive(&producer_mbar[write_pipe]);
            ++write_state;
            --k_tile_count;
            load_tile += gridDim.x;
        }
```

对于消费者而言，其实也是类似的，使用一个read_state去标记phase和所处的阶段。使用`ProducerBarType`去等待当前stage的写入，当到达后使用`ConsumerBarType`去arrive

```cpp
        int consumer_tid = tid - PRODUCER_THREADS;  // 本地索引 0..95
        auto read_state = cutlass::PipelineState<PIPE_STAGES>();
        int k_tile_count = my_tile_count;
        int store_tile = blockIdx.x;

        // ---- 主循环：等待 Producer 填充数据 → 计算 ×2 → 写回 GMEM → 释放 buffer ----
        CUTE_NO_UNROLL
        while (k_tile_count > 0) {
            int read_pipe = read_state.index();

            // 等待 Producer 完成当前 stage 的数据加载
            // mbarrier::wait 阻塞直到 32 个 Producer 线程都已 arrive 且 phase 匹配
            ProducerBarType::wait(&producer_mbar[read_pipe], read_state.phase());

            // 96 个线程以 stride 方式处理 128 个元素，部分线程处理 2 个元素
            int global_offset = store_tile * TILE_SIZE;
            for (int i = consumer_tid; i < TILE_SIZE; i += CONSUMER_THREADS) {
                float val = compute_func(smem.buf[read_pipe][i]);
                if (global_offset + i < N) {
                    out[global_offset + i] = val;
                }
            }

            // 所有 96 个 Consumer 线程 arrive，通知 Producer "buffer 已消费完毕，可覆写"
            ConsumerBarType::arrive(&consumer_mbar[read_pipe]);
            ++read_state;
            --k_tile_count;
            store_tile += gridDim.x;
        }
```