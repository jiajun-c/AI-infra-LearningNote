# Slime 框架

## 1. 初始化

主要是分配每个卡做训/推，如果是在一张卡上做训/推的话，那么会把40%的分别分配给训/推，剩下一些显存

```python
# placement_group.py:79-119
def create_placement_groups(args):
    # 非 colocate 模式（本脚本）：
    # actor 4 GPU (indices 0-3) + rollout 4 GPU (indices 4-7)
    num_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node  # 4
              + args.rollout_num_gpus                                # 4
    rollout_offset = args.actor_num_nodes * args.actor_num_gpus_per_node  # =4

    pg, bundle_indices, gpu_ids = _create_placement_group(num_gpus)  # 申请8块GPU

    # 按 GPU 物理编号排序后切片
    rollout_pg = bundle_indices[rollout_offset:]   # GPU 4-7 → SGLang
    actor_pg   = bundle_indices[:rollout_offset]   # GPU 0-3 → Megatron
```

## 2. 主循环

先生产rollout数据，然后根据rollout产生的数据去做RL

```python
# train.py:73-102
for rollout_id in range(args.start_rollout_id, args.num_rollout):  # 0 → 2999

    # ① 先跑 eval（仅 rollout_id==0 时，训练前基线）
    if args.eval_interval is not None and rollout_id == 0:
        ray.get(rollout_manager.eval.remote(rollout_id))

    # ② 生成 rollout 数据（阻塞，等 SGLang 推理完成）
    rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))

    # ③ 训练 actor（阻塞，等 Megatron 训练完成）
    ray.get(actor_model.async_train(rollout_id, rollout_data_ref))

    # ④ 每 20 步 save + eval
    if should_run_periodic_action(rollout_id, args.save_interval, ...):
        save(rollout_id)   # → /root/GLM-Z1-9B-0414_slime/

    # ⑤ 同步权重到 SGLang
    actor_model.update_weights()

    if should_run_periodic_action(rollout_id, args.eval_interval, ...):
        ray.get(rollout_manager.eval.remote(rollout_id))
```

## 3. RolloutManager.generate()

调用Rollout生成样本，然后转化为训练的dict，

```python
# rollout.py:479-492
def generate(self, rollout_id):
    # 1. 调用 sglang_rollout.py 的 generate_rollout() 生成样本
    data, metrics = self._get_rollout_data(rollout_id=rollout_id)

    # 2. 将 Sample 列表转成训练用的 dict
    data = self._convert_samples_to_train_data(data)

    # 3. 按 DP rank 切片，存入 Ray object store，返回 refs
    return self._split_train_data_by_dp(data, self.train_parallel_config["dp_size"])
```

将sampler对象展平

```python
train_data = {
    "tokens":           [sample.tokens ...],           # prompt+response token ids
    "response_lengths": [sample.response_length ...],  # response 部分长度
    "rewards":          rewards,                       # 后处理后的 reward（GRPO 归一化后）
    "raw_reward":       raw_rewards,                   # deepscaler 原始 0/1
    "loss_masks":       loss_masks,                    # 1=response token, 0=prompt token
    "truncated":        [...],
}
```
再按照DP给均分这个样本

## 4. Reward 计算

计算reward

```python
def get_deepscaler_rule_based_reward(response, label):
    # 提取 </think> 之后的内容作为答案区域
    if "</think>" in response:
        model_solution = response.split("</think>")[-1]
    elif "###Response" in response:
        model_solution = response.split("###Response")[1]
    else:
        return 0   # 没有 </think> → 直接 0 分

    model_answer = extract_answer(model_solution)   # 从 \boxed{} 提取答案
    if model_answer is None:
        return 0

    # 数学等价判断（两种方式）
    for ground_truth in processed_ground_truths:
        if grade_answer_mathd(model_answer, ground_truth) \
           or grade_answer_sympy(model_answer, ground_truth):
            return 1   # 正确

    return 0   # 错误
```

## 5. Actor训练

使用ref_model进行前向，计算KL loss，切回actor model计算log_probs，

```python
# actor.py:406-490
def train_actor(self, rollout_id, rollout_data):
    data_iterator, num_microbatches = get_data_iterator(...)

    with timer("train"):
        # ① 用 ref model 前向，计算 ref_log_probs（用于 KL loss）
        self._switch_model("ref")
        rollout_data.update(self.compute_log_prob(..., store_prefix="ref_"))
        #   → actor.py:353: forward_only(get_log_probs_and_entropy, ...)
        #   → 耗时 ~41s（来自 log）

        # ② 切回 actor model，计算当前 log_probs
        self._switch_model("actor")
        rollout_data.update(self.compute_log_prob(...))
        #   → 耗时 ~41s

        # ③ 计算 GRPO advantages
        compute_advantages_and_returns(self.args, rollout_data)

        # ④ 反向传播训练
        with timer("actor_train"):
            train(rollout_id, self.model, self.optimizer, ...)

    # ⑤ 备份最新权重到 CPU，供后续同步到 SGLang
    self.weights_backuper.backup("actor")
```

## 6. GRPO Advantage 计算

```python
# loss.py:619-623
if args.advantage_estimator in ["grpo", "gspo"]:
    rewards = torch.tensor(rewards, ...)   # shape: [256]，已经是组内归一化后的值

    # get_grpo_returns: 每个 response token 都用相同的 reward 值
    returns = get_grpo_returns(rewards, kl)
    advantages = [r for r in returns]     # advantages[i] 是一个全为 reward_i 的向量

# ppo_utils.py:201-208
def get_grpo_returns(rewards, kl):
    returns = []
    for i in range(len(rewards)):
        # 每个 token 的 return = 该 response 的 reward（标量广播）
        returns.append(torch.ones_like(kl[i]) * rewards[i])
    return returns
```