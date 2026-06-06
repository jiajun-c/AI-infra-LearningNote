# 对齐

让模型的输出更符合人类偏好(有用，无害，诚实)

- RLHF(经典三步)
  - 1, 训练奖励模型 RM：人类对多个回答排序-> 学习打分
  - 2，PPO强化学习，用RN的分数作为奖励信号优化模型

- DPO(direct preference optimization)
  跳过奖励模型，直接使用偏好模型去优化策略模型
  训练数据
  - 指令x
  - 好的回答 y_w(chosen)
  - 坏的回答 y_l(rejected)
