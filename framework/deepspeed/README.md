# Deepspeed 入门

deepspeed 是掩盖微软开发的开源深度学习优化库，用于提升大规模模型训练和推理的效率

## 1. 初始化

初始化`deepspeed`
```python3
model_engine, optimizer, trainloader, _ = deepspeed.initialize(
    args=args,
    model=net,
    model_parameters=parameters,
    training_data=trainset,
)
```

在分布式环境下，需要进行如下所示的初始化

```python3
deepspeed.init_distributed()
```

## 2. 训练

完成deepspeed 引起的初始化后，可以基于其进行训练。

```python3
for step, batch in enumerate(data_loader):
    #forward() method
    loss = model_engine(batch)

    #runs backpropagation
    model_engine.backward(loss)

    #weight update
    model_engine.step()
```

## 3. deepspeed 配置

deepspeed 可以使用一个json文件进行配置如下所示，如下所示使用fp16进行训练，同时使用Adam优化器进行训练。

```json
{
  "train_batch_size": 8,
  "gradient_accumulation_steps": 1,
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 0.00015
    }
  },
  "fp16": {
    "enabled": true
  },
  "zero_optimization": true
}
```