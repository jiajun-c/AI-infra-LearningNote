# vllm 代码梳理

## 1. 整体框架

- engine: 核心调度器和引擎入口
- model_executor: 模型定义与执行
- config: 系统配置中心
- compilation: 图编译优化
- kernels：高性能算子接口
- device_allocator: 底层显存分配器
- platforms: 异构硬件适配层
- distributed: 分布式原语层
- entrypoints：API与输入输出处理
....

## 2. 核心框架

最核心的代码其实在`vllm/vllm/v1`，其中存放了关于attention实现，调度算法

### 2.1 vllm调度算法

在调度算法开始时会设置限制条件
- max_num_running_reqs: 一次前向传播中调度器最多能同时处理的独立序列
- max_num_scheduled_tokens: 单步最大调度token数
- max_model_len: 模型最大上下文长度

```shell
        self.max_num_running_reqs = self.scheduler_config.max_num_seqs
        self.max_num_scheduled_tokens = (
            self.scheduler_config.max_num_scheduled_tokens
            if self.scheduler_config.max_num_scheduled_tokens
            else self.scheduler_config.max_num_batched_tokens
        )
        self.max_model_len = vllm_config.model_config.max_model_len
```

任务队列信息
- self.waiting: 等待队列
- self.skipped_waiting: 跳过队列，本来应该执行但是暂时无法被调度
- self.running: 运行列表，正在被处理的请求
- self.finished_req_ids: 完结ID集合，需要回收显存
- self.num_waiting_for_streaming_input: 流式输入等待计数器，流式输入交互，等待客户端新的输入

```shell
        self.waiting = create_request_queue(self.policy)
        # requests skipped in waiting flow due async deps or constraints.
        self.skipped_waiting = create_request_queue(self.policy)
        self.running: list[Request] = []

        # The request IDs that are finished in between the previous and the
        # current steps. This is used to notify the workers about the finished
        # requests so that they can free the cached states for those requests.
        # This is flushed at the end of each scheduling step.
        self.finished_req_ids: set[str] = set()

        # Counter for requests waiting for streaming input. Used to calculate
        # number of unfinished requests
        self.num_waiting_for_streaming_input: int = 0
```

进行调度的时候将会调用`schedule`函数, vllm的schedule其实是不区分P/D阶段的，

```python
    def schedule(self) -> SchedulerOutput:
        # NOTE(woosuk) on the scheduling algorithm:
        # There's no "decoding phase" nor "prefill phase" in the scheduler.
        # Each request just has the num_computed_tokens and
        # num_tokens_with_spec. num_tokens_with_spec =
        # len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).
        # At each step, the scheduler tries to assign tokens to the requests
        # so that each request's num_computed_tokens can catch up its
        # num_tokens_with_spec. This is general enough to cover
        # chunked prefills, prefix caching, speculative decoding,
        # and the "jump decoding" optimization in the future.

        scheduled_new_reqs: list[Request] = []
        scheduled_resumed_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []
        preempted_reqs: list[Request] = []
```

在每次调度的时候会初始化几个队列

- scheduled_new_reqs: 新请求/从waiting队列里面被捞出来的请求
- scheduled_resumed_reqs: 将原来被换出的请求替换回来
- scheduled_running_reqs: 上一个step在running队列的，并且这一步依然有足够显存让他们继续运行
- preempted_reqs：惨遭抢占的请求

其会首先遍历`running`队列，同时不断查看是否还有剩余的预算

`num_output_placeholders` 大于0的时候表示这启用了投机推理，如果此时达到了请求的最大长度，那么放弃当前请求

```python
            if (
                request.num_output_placeholders > 0
                and request.num_computed_tokens + 2 - request.num_output_placeholders
                >= request.num_prompt_tokens + request.max_tokens
            ):
                req_index += 1
                continue
```

然后计算新产生的token(包含投机推理)

```python
            num_new_tokens = (
                request.num_tokens_with_spec
                + request.num_output_placeholders
                - request.num_computed_tokens
            )
```

这个token数量可能会超过我们在不同层面的预算

- 显存层面：可能会超过我们的`token_budget`，num_new_tokens = min(num_new_tokens, token_budget)
- chunked prefill限制：控制其不要超过chunked prefill的长度限制
- 最长上下文限制：根据模型上下文来限制本次产生的量

```python3
            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold
            num_new_tokens = min(num_new_tokens, token_budget)

            # Make sure the input position does not exceed the max model len.
            # This is necessary when using spec decoding.
            num_new_tokens = min(
                num_new_tokens, self.max_model_len - 1 - request.num_computed_tokens
            )
```

如果此时没有新的token产生了(max_new_token == 0)，那么跳过当前的请求。

如果处理了当前的请求，那么将会进行资源分配的阶段，对于资源分配的阶段其实也分为两条路径

- `kv_cache_manager`可以成功分配所需的空间
- `kv_cache_manager`已满，需要驱逐之前的请求释放空间

```python
                while True:
                    new_blocks = self.kv_cache_manager.allocate_slots(
                        request,
                        num_new_tokens,
                        num_lookahead_tokens=self.num_lookahead_tokens,
                    )

                    if new_blocks is not None:
                        # The request can be scheduled.
                        break
```

被抢占的逻辑如下所示，会去找到其中优先级最低的，将其抢占，如果没有抢占到，那么本次调度将会先跳过，抢占的情况下将被
抢占请求加入到`preempted_reqs`中

```python
                    if self.policy == SchedulingPolicy.PRIORITY:
                        preempted_req = max(
                            self.running,
                            key=lambda r: (r.priority, r.arrival_time),
                        )
                        self.running.remove(preempted_req)
                        if preempted_req in scheduled_running_reqs:
                            preempted_req_id = preempted_req.request_id
                            scheduled_running_reqs.remove(preempted_req)
                            token_budget += num_scheduled_tokens.pop(preempted_req_id)
                            req_to_new_blocks.pop(preempted_req_id)
                            scheduled_spec_decode_tokens.pop(preempted_req_id, None)
                            preempted_encoder_inputs = scheduled_encoder_inputs.pop(
                                preempted_req_id, None
                            )
                            if preempted_encoder_inputs:
                                # Restore encoder compute budget if the preempted
                                # request had encoder inputs scheduled in this step.
                                num_embeds_to_restore = sum(
                                    preempted_req.get_num_encoder_embeds(i)
                                    for i in preempted_encoder_inputs
                                )
                                encoder_compute_budget += num_embeds_to_restore
                            req_index -= 1
                    else:
                        preempted_req = self.running.pop()

                    self._preempt_request(preempted_req, scheduled_timestamp)
                    preempted_reqs.append(preempted_req)
                    if preempted_req == request:
                        # No more request to preempt. Cannot schedule this request.
                        break
```

成功调度后分配对应的空间，调整token_budget

```python
            scheduled_running_reqs.append(request)
            request_id = request.request_id
            req_to_new_blocks[request_id] = new_blocks
            num_scheduled_tokens[request_id] = num_new_tokens
            token_budget -= num_new_tokens
            req_index += 1
```

假设不需要抢占请求那么说明现在的系统处于一个比较空闲的状态可以调度waiting队列的请求

