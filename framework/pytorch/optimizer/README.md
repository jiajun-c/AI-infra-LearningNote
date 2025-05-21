# Torch 优化器

## 1. Torch中的优化器基类

Torch中有多种优化器如SGD，Adam，AdamW之类的，他们都继承于`torch.optim.Optimizer` 这一个基类。其中定义了一些常用的方法，如`zero_grad`, `step(closure)`, `state_dict`, `load_state_dict`等。

Optimeizer 中有三个属性
- defaults: 存储的是优化器的超参数，例子如下
- state：参数的缓存
- param_groups：参数组，每一个参数组都是一个字典，存储的是参数的优化器超参数，如params(从模型中提取的某一组网络层的权重和偏置), lr，momentum，weight_decay等

其中还有下面的一些函数
- zero_grad()：清空梯度
- step()：更新参数
- add_param_group()：添加参数组
- load_state_dict()：加载优化器的状态
- state_dict()：保存优化器的状态

## 2. 常见优化器

