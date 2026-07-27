# 一个训练的Step组成

## 1. 显存组成

- Master weight：Fp32，优化器真正更新的对象
- Model weight：Bf16，前向/反向计算用，是master weight的cast的副本
- 梯度g：BF16（累加的时候使用fp32），是反向的产出
- Adam一阶钜m FP32, 优化器状态
- Adma二阶钜v FP32, 优化器状态

## 2. 梯度累加

全局batch（比如8M tokens）远超单卡单次能跑，所以要去切分 `mico_batch`

mico_batch 内的不需要梯度的累加，仅在最后一个step进行优化器的更新

## 3. 梯度同步

DDP(朴素数据并行)

- 反向过程中，梯度按bucket就绪一组，通信和反向计算重叠，而不是等整个backward结束再通信
- all-reduce做的是求平均

zero-2/FSDP

- 不做all-reduce，做reduce-scatter，每张卡只收到自己负责1/N参数的完整梯度，因为优化器的状态也是切分的
- zero-2/FSDP 额外的前向/反向前all-gather，用完即释放

## 4. 梯度裁剪

在`optimzer.step()`之前执行，必须在梯度同步完成后，基于全局梯度做

- 计算全局L2范数
- 若total_norm > max_norm，所有梯度max_norm/(total_norm + 1e-6)
- grad_norm 这个值需要进行监控，假设突然进行飙升，可能出现loss spike

## 5. AdamW更新

对每个参数(在FP32 master weight上执行更新的步骤)

## 6. 学习率调度

lr不是一个常数，每个step由scheduler算出塞入到optimzer
