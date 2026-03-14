# CUDA bench and Profile

## 1. 在Pytorch中的计时

简单地使用pytorch中地cuda event进行计时，由于cuda是异步的，所以不可以使用python的计时模块

```python3
def time_torch_function(func, input):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(5):
        func(input)
    start.record()
    func(input)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)
```

使用torch的profile工具，再根据cuda运行的整体时间进行排序

```pytorch
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    torch.square(b)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

```

