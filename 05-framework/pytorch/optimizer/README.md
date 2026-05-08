# Torch 优化器

## 1. Torch中的优化器基类

Torch中有多种优化器如SGD，Adam，AdamW之类的，他们都继承于`torch.optim.Optimizer` 这一个基类。其中定义了一些常用的方法，如`zero_grad`, `step(closure)`, `state_dict`, `load_state_dict`等。

Optimizer 中有三个属性
- defaults: 存储的是优化器的超参数，例子如下
- state：参数的缓存，如动量、梯度平方均值等中间状态
- param_groups：参数组，每一个参数组都是一个字典，存储的是参数的优化器超参数，如params(从模型中提取的某一组网络层的权重和偏置), lr，momentum，weight_decay等

其中还有下面的一些函数
- zero_grad()：清空梯度
- step()：更新参数
- add_param_group()：添加参数组
- load_state_dict()：加载优化器的状态（用于断点续训）
- state_dict()：保存优化器的状态

param_groups 的意义：可以对不同层设置不同超参数，例如：

```python
optimizer = torch.optim.Adam([
    {'params': model.encoder.parameters(), 'lr': 1e-4},
    {'params': model.decoder.parameters(), 'lr': 1e-3}
])
```

## 2. 常见优化器

优化器的演进路线：

```
SGD
 ├── + 动量 → SGD with Momentum
 │
 └── + 自适应学习率
      ├── AdaGrad（累积历史梯度平方）
      ├── RMSProp（指数移动平均替代累积）
      └── Adam（动量 + RMSProp）
           └── AdamW（Adam + 解耦 weight decay）
```

### 2.1 SGD

其中 $\theta_{t}$ 是参数值，$\eta$ 是学习率，其问题在于学习率固定，一刀切的效果差

$$\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta L$$

### 2.2 SGD with Momentum

给更新加上了一个惯性，$v_{t-1}$ 中累积了过去梯度的方向，减少震荡，加速收敛。直觉上像小球滚山坡，有惯性，不容易被小扰动影响。

$$v_t = \beta \cdot v_{t-1} + \nabla_\theta L$$
$$\theta_{t+1} = \theta_t - \eta \cdot v_t$$

$\beta$ 通常取 0.9，相当于过去梯度的加权平均。

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

### 2.3 AdaGrad

稀疏特征（如 NLP 中的词嵌入）不同参数梯度差异极大，需要自适应学习率。

$$G_t = G_{t-1} + g_t^2$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \cdot g_t$$

$G_t$ 累积每个参数的历史梯度平方和，梯度大的参数学习率自动缩小，梯度小的参数学习率保持相对大。

缺点：$G_t$ 单调递增，学习率会持续缩小直到趋近于 0，训练后期几乎停止更新。

### 2.4 RMSProp

解决 AdaGrad 学习率消亡问题，用**指数移动平均**替代累积，只关注近期梯度。

$$v_t = \beta \cdot v_{t-1} + (1-\beta) \cdot g_t^2$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{v_t + \epsilon}} \cdot g_t$$

$\beta$ 通常取 0.9。

```python
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001, alpha=0.9)
```

### 2.5 Adam（Adaptive Moment Estimation）

结合 Momentum（一阶矩）和 RMSProp（二阶矩）的优点。

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \quad \text{（一阶矩，动量）}$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \quad \text{（二阶矩，梯度平方均值）}$$

初始阶段 $m_t, v_t$ 偏小，需要偏差修正：

$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \cdot \hat{m}_t$$

默认超参数：$\beta_1=0.9,\ \beta_2=0.999,\ \epsilon=10^{-8}$，每个参数有独立的自适应学习率。

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
```

### 2.6 AdamW（解耦 Weight Decay）

**问题**：Adam 中 L2 正则化和 weight decay 并不等价。在 Adam 中，梯度 $\nabla L + \lambda \theta_t$ 会被自适应学习率缩放，导致 weight decay 的实际效果被梯度大小干扰。

**AdamW 的修复**：将 weight decay 从梯度中解耦，直接作用在参数上：

$$\theta_{t+1} = (1-\eta\lambda)\theta_t - \frac{\eta}{\sqrt{\hat{v}_t}+\epsilon} \hat{m}_t$$

现代 LLM 训练（GPT、LLaMA 等）几乎都使用 AdamW。

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
```

## 3. 根据不同场景选择不同的优化器

| 场景 | 推荐优化器 |
|------|-----------|
| 计算机视觉（CNN） | SGD + Momentum（收敛更好，泛化略优） |
| NLP / Transformer | AdamW |
| 稀疏特征（推荐系统） | Adagrad / Adam |
| 快速调参/验证 | Adam |
| 大模型预训练 | AdamW（+ 学习率调度） |

## 4. 学习率调度（Scheduler）

优化器通常配合学习率调度一起使用：

```python
# Warmup + Cosine Decay（LLM 训练标配）
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)

# 训练循环
for batch in dataloader:
    loss = model(batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
```

## 5. 自定义优化器

理解优化器最好的方式是动手实现，以下是一个带动量的 SGD 实现：

```python
class MySGD(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0):
        defaults = {'lr': lr, 'momentum': momentum}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                if momentum > 0:
                    state = self.state[p]
                    if 'v' not in state:
                        state['v'] = torch.zeros_like(p)
                    v = state['v']
                    v.mul_(momentum).add_(g)  # v = momentum*v + g
                    g = v
                p.add_(g, alpha=-lr)          # p = p - lr * g
```
