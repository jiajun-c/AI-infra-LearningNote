# vLLM并行策略

并行的整体布局为 [ExternalDP, DP, PP, TP]

ExternalDP是外部的并行，每个实例单独执行，他们之间不会有通信/同步

## 1. TP

TP并行是最基础的并行策略，其初始化代码如下

```python
    all_ranks = torch.arange(world_size).reshape(
        -1,
        data_parallel_size,
        pipeline_model_parallel_size,
        prefill_context_model_parallel_size,
        tensor_model_parallel_size,
    )  # noqa

    # Build the tensor model-parallel groups.
    global _TP
    assert _TP is None, "tensor model parallel group is already initialized"
    group_ranks = all_ranks.view(-1, tensor_model_parallel_size).unbind(0)
    group_ranks = [x.tolist() for x in group_ranks]
    if enable_elastic_ep:
        group_ranks = local_all_ranks.view(-1, tensor_model_parallel_size).unbind(0)
        group_ranks = [x.tolist() for x in group_ranks]
    # message queue broadcaster is only used in tensor model parallel group
    _TP = init_model_parallel_group(
        group_ranks,
        get_world_group().local_rank,
        backend,
        use_message_queue_broadcaster=True,
        group_name="tp",
    )
```

进行`forward`的代码如下所示，按列切分到卡上，然后先计算矩阵乘法然后通过一次`all_gather`获取到全部的结果。

### 1.1 列并行

对权重在列上进行切分

```python
    def forward(
        self,
        input_,
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        bias = self.bias if not self.skip_bias_add else None

        # Matrix multiply.
        assert self.quant_method is not None
        output_parallel = self.quant_method.apply(self, input_, bias)

        if self.gather_output and self.tp_size > 1:
            # All-gather across the partitions.
            output = tensor_model_parallel_all_gather(output_parallel)
        else:
            output = output_parallel

        if not self.return_bias:
            return output
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias
```

### 1.2 行并行

对权重在行上进行切分

```python
    def forward(
        self,
        input_,
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        if self.input_is_parallel:
            input_parallel = input_
        else:
            split_input = split_tensor_along_last_dim(
                input_, num_partitions=self.tp_size
            )
            input_parallel = split_input[self.tp_rank].contiguous()

        # Matrix multiply.
        assert self.quant_method is not None
        # Only fuse bias add into GEMM for rank 0 (this ensures that
        # bias will not get added more than once in TP>1 case)
        bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
        output_parallel = self.quant_method.apply(self, input_parallel, bias_)

        if self.reduce_results and self.tp_size > 1:
            output = tensor_model_parallel_all_reduce(output_parallel)
        else:
            output = output_parallel

        if not self.return_bias:
            return output
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias
```

## 2. PP

PP并行是将模型不同层的权重切分到不同的卡上

### 2.1 权重切分

如下所示，为每个组划分他自己拥有的层，其他层使用`PPMissingLayer`进行填充，如下所示

```python
    from vllm.distributed.parallel_state import get_pp_group
    from vllm.distributed.utils import get_pp_indices
    from vllm.model_executor.offloader import get_offloader

    start_layer, end_layer = get_pp_indices(
        num_hidden_layers, get_pp_group().rank_in_group, get_pp_group().world_size
    )

    modules = torch.nn.ModuleList(
        [PPMissingLayer() for _ in range(start_layer)]
        + get_offloader().wrap_modules(
            layer_fn(prefix=f"{prefix}.{idx}") for idx in range(start_layer, end_layer)
        )
        + [PPMissingLayer() for _ in range(end_layer, num_hidden_layers)]
    )
```


```python
    global _PP
    assert _PP is None, "pipeline model parallel group is already initialized"
    group_ranks = (
        all_ranks.transpose(2, 4).reshape(-1, pipeline_model_parallel_size).unbind(0)
    )
    group_ranks = [x.tolist() for x in group_ranks]
    if enable_elastic_ep:
        group_ranks = (
            local_all_ranks.transpose(0, 2)
            .reshape(-1, pipeline_model_parallel_size)
            .unbind(0)
        )
        group_ranks = [x.tolist() for x in group_ranks]
    _PP = init_model_parallel_group(
        group_ranks, get_world_group().local_rank, backend, group_name="pp"
    )
```

TP和PP本质上都是对模型权重进行了一个切分，只是通信的方式不一样，做TP之后要去做allgather，但是PP只需要在两个相邻的节点之间去做点对点通信即可

## 3. DP

初始化部分

```python
    global _DP
    assert _DP is None, "data parallel group is already initialized"
    group_ranks = all_ranks.transpose(1, 4).reshape(-1, data_parallel_size).unbind(0)
    group_ranks = [x.tolist() for x in group_ranks]
    if enable_elastic_ep:
        _DP = _init_stateless_group(
            group_ranks,
            "dp",
            parallel_config.data_parallel_master_ip,
            backend,
            coord_store=coord_store,
        )
    else:
        _DP = init_model_parallel_group(
            group_ranks, get_world_group().local_rank, backend, group_name="dp"
        )
```

DP并行有三种方式
- internal LB：默认，每个vllm进程，管理全部的DP rank，vllm自己做负载均衡
- External LB：每个rank自己单独起一个vllm serve，负载均衡完全交给外部去做
- Hybrid LB：每个节点自己起一个vllm serve管理自己的subset，然后节点之间进行External LB

## 4. CP

### 4.1 DCP

DCP是把TP重新再做一次切分把他切分到DCP

每个 DCP rank 持有 KV cache 的一段（沿 seq_len 维度分片）。Decode 时每个 rank 只看到部分历史 token，需要聚合所有 rank 的局部 attention 输出。

1. AllGather(query, dim=1)
   → 每个 rank 获得全部 head 的 query

2. 本地计算 attention（只用本 rank 的 KV 分片）
   → partial_output [B, H, D], partial_lse [B, H]

3. 交换各 rank 的 partial_output 和 partial_lse
   → 两种后端可选

4. LSE 加权合并（Triton kernel）
   → global_output = Σ output_i × exp(lse_i - lse_max) / Σ exp(lse_j - lse_max)

5. ReduceScatter(corrected_output, dim=head)
   → 每个 rank 得到自己负责的 head 分片

在没有DCP的情况下，每张卡上都会存一份KV cache

有DCP的情况下，每张卡只存自己的那部分？对前缀匹配的影响？

### 4.2 PCP

PCP是把Prefill部分切分到多卡，分为两种策略

- 序列适中的时候，每个卡算自己的那部分query，但是kv是各自共享的
- 长序列的时候，每个卡算自己的那部分query，同时KV也会进行切分，通过点对点通信传递 (ring Attention)

### 4.3 DCP + PCP

decode和prefill阶段的特性不同，prefill阶段需要保证尽量减少通信，本身就是一个计算bound，

而decode阶段是一个显存bound，可以考虑通过更大的dcp size来减轻单卡的显存压力

### 4.4 CP并行是如何减轻kvcache压力

会先分为一个逻辑kv cache，然后会去判断哪些block是真的属于本地的，token 按条带轮询分配给各 rank，rank r 只写入 token_pos % N == r 的 token，其余位置为 PAD，不占实际 KV Cache 存储。

## 5. EP

一个EP组内包括 DP * PCP * TP,EP组的排布方式上有两种
- 线性排列：连续的几个专家会被分配到一个卡上
- roundRobin：轮询地分配

```python
    global _EP
    assert _EP is None, "expert parallel group is already initialized"
    # Don't create EP group for dense models.
    if config.model_config is None or config.model_config.is_moe:
        group_ranks = (
            all_ranks.transpose(1, 2)
            .reshape(
                -1,
                data_parallel_size
                * prefill_context_model_parallel_size
                * tensor_model_parallel_size,
            )
            .unbind(0)
        )
```

## 6. EPLB

负载均衡的专家并行，对于一些热点的专家，为其创建冗余的专家(逻辑上)，权重也需要通过通信进行传递

