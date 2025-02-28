# Triton 自动调优

Triton中可以使用autotune装饰器对列表中的参数进行自动地调优

```python3
triton.autotune(configs,
                key, 
                prune_configs_by=None, 
                reset_to_zero=None, 
                restore_value=None, 
                pre_hook=None, 
                post_hook=None, 
                warmup=25, 
                rep=100, 
                use_cuda_graph=False)
```

列表成员的详细信息如下

https://triton.hyper.ai/docs/python-api/triton/triton_autotune/

如下是一个例子，当x_size 发生变化的适合会重新计算最优的参数


```python3
@triton.autotune(configs=[
    triton.Config(kwargs={'BLOCK_SIZE': 128}, num_warps=4),
    triton.Config(kwargs={'BLOCK_SIZE': 1024}, num_warps=8),
  ],
  key=['x_size'] # the two above configs will be evaluated anytime 
                 # the value of x_size changes  
)
@triton.jit
def kernel(x_ptr, x_size, **META):
    BLOCK_SIZE = META['BLOCK_SIZE']
```
