# 学习率调度器

## 1. 余弦学习率调度器

余弦衰减公式

$\eta_{t} = \eta_{min}+\frac{1}{2}(\eta_{max}-\eta_{min})(1 + \cos(\frac{\pi t}{T_{max}}))$

- $\eta_{max}$: 最大学习率
- $\eta_{min}$: 最小学习率
- $T_{max}$ 每个周期的总步数
- t：当前的步数

函数特性
- 训练初期使用该函数可以使得学习率从 $\eta_{max}$ 快速下降
- 训练后期可以使得学习率接近$\eta_{min}$
- 周期性重启：每个周期结束时重置学习率到初始值

代码实现
在warmup阶段使用线性lr，在warmup结束之后使用cosine lr
```python
def get_cosine_lr(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int
    ):
    if it < warmup_iters:
        ans = max_learning_rate * it / warmup_iters
    elif it <= cosine_cycle_iters:
        ans = min_learning_rate + 0.5 * (max_learning_rate - min_learning_rate) * (1 + math.cos(math.pi * it / cosine_cycle_iters))
    else:
        ans = min_learning_rate
    return ans
```