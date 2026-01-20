# torch 装饰器

torch中的装饰器可以用来控制上下文/改变执行行为

## torch.no_grad

关闭梯度计算，但是不关闭视图跟踪

## torch.enable_grad

开启梯度计算

## torch.compile

启用jit编译，分析代码逻辑，将其编译为优化的Triton kernel

## torch.inference_mode

用于关闭梯度计算，系统不再记录操作历史，和torch.no_grad()一样，同时不需要跟踪张量的版本计数器，即使原地修改变量也不会出现报错