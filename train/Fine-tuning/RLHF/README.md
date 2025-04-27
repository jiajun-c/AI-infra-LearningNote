# RLHF(Reinforcement Learning from Human Feedback)

RLHF的目标是通过人类反馈信号调整模型的行为。收集人类的偏好根据设计奖励函数。



RLHF的流程如下
- 多种策略产生样本并收集人类反馈
- 训练奖励模型
- 训练强化学习策略，微调LM


奖励函数如下所示

![alt text](image-1.png)
