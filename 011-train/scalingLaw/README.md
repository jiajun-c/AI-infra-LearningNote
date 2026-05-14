# Scaling Law

## Chinchilla Scaling Law / Compute-Optimal Scaling Law

论文：[Training Compute-Optimal Large Language Models](./train-compute.pdf)

在之前的论文中，提出了 Scaling Law，但是之前的文章侧重点在参数层面，认为参数的增加是更加重要的。但是 Chinchilla 这篇文章提出：在固定训练算力预算下，数据和参数增长的比例应该接近一样。

Chinchilla 训练了一个 70B 的模型，但是使用了 1.4T tokens 进行训练，结果其效果超过了 280B 参数、300B tokens 的 Gopher。

这篇论文研究的问题是：

```text
给定固定训练算力 C，应该选择多大的模型参数量 N，
以及训练多少 token D，才能让最终 loss 最低？
```

论文把目标写成：

```text
N_opt(C), D_opt(C) = argmin L(N, D)
                    s.t. FLOPs(N, D) = C
```

其中：

```text
N: 模型参数量
D: 训练 token 数
C: 训练 FLOPs 预算
L(N, D): 最终训练 loss
```

论文最后得到的核心结论是：

```text
N_opt ∝ C^0.5
D_opt ∝ C^0.5
```

也就是说，当训练算力增加时，模型参数量和训练 token 数应该大致同比例增加。

## 三种估计方法

### 方法 1：固定模型大小，改变训练 token 数

第一种方法是固定一组模型参数量，然后对每个模型训练不同长度。

比如：

```text
N 固定
D 变化
观察不同训练 token 下的 loss 曲线
```

论文训练了从 70M 到 10B 参数的模型，每个模型配不同的 cosine learning rate schedule 和训练 token 数。

然后它把所有训练曲线放在一起，做一件事：

```text
对于每一个 FLOPs 预算 C，
找出所有训练曲线里 loss 最低的点。
```

这些最低点形成了一条 compute-efficient envelope，也就是在不同算力预算下的最优边界。

最后对这些最优点拟合幂律：

```text
N_opt ∝ C^a
D_opt ∝ C^b
```

方法 1 得到：

```text
a = 0.50
b = 0.50
```

直观理解：

```text
如果我有这么多训练算力，
在所有试过的模型和训练长度里，哪个组合最划算？
```

### 方法 2：固定 FLOPs，改变模型大小

第二种方法叫 IsoFLOP profiles。

它先固定一个训练算力预算 C，然后训练不同大小的模型。因为训练 FLOPs 近似满足：

```text
C ≈ 6ND
```

所以当 C 固定时：

```text
模型越大 N 越大，可以训练的 token D 就越少
模型越小 N 越小，可以训练的 token D 就越多
```

然后比较这些模型的最终 loss。

论文发现，在同一个 FLOPs 预算下，loss 和模型大小之间会形成一个谷底：

```text
模型太小：容量不够，loss 高
模型太大：训练 token 不够，loss 也高
中间某个大小：loss 最低
```

这个谷底对应的模型参数量，就是当前算力预算下的最优 N_opt。

论文对多个 FLOPs 预算重复这个过程，再拟合：

```text
N_opt ∝ C^a
D_opt ∝ C^b
```

方法 2 得到：

```text
a = 0.49
b = 0.51
```

直观理解：

```text
同样花这么多 FLOPs，
到底是训练小模型很久，还是训练大模型较短时间？
哪个最终 loss 更低？
```

### 方法 3：拟合参数化 loss 函数

第三种方法更加理论化。论文假设最终 loss 可以写成：

```text
L(N, D) = E + A / N^alpha + B / D^beta
```

其中：

```text
E: 数据分布本身的不可约 loss
A / N^alpha: 模型参数量不够带来的误差
B / D^beta: 训练 token 不够带来的误差
```

然后用所有实验结果拟合这些参数：

```text
A, B, E, alpha, beta
```

拟合完成后，在 FLOPs 约束下求最优解：

```text
C ≈ 6ND
```

也就是在固定算力预算 C 的条件下，寻找让 L(N, D) 最小的 N 和 D。

方法 3 得到：

```text
a = 0.46
b = 0.54
```

这个结果仍然接近 1:1，只是稍微更偏向增加训练数据。

## 三种方法的结论对比

| 方法 | N_opt 的指数 a | D_opt 的指数 b | 含义 |
| --- | --- | --- | --- |
| 方法 1：训练曲线 envelope | 0.50 | 0.50 | 参数和数据等比例增长 |
| 方法 2：IsoFLOP profiles | 0.49 | 0.51 | 参数和数据几乎等比例增长 |
| 方法 3：参数化 loss 拟合 | 0.46 | 0.54 | 稍微更偏向增加数据 |
| Kaplan et al. | 0.73 | 0.27 | 更偏向增加参数 |

所以 Chinchilla Scaling Law 的关键修正是：

```text
之前：更多算力主要应该用来扩大模型参数量
现在：更多算力应该同时扩大模型参数量和训练数据量
```

一个常用的经验记忆是：

```text
compute-optimal training 中，每 1 个参数大约对应 20 个训练 token
```

例如：

```text
70B 参数 × 20 ≈ 1.4T tokens
```

这正好对应 Chinchilla 的训练配置。
