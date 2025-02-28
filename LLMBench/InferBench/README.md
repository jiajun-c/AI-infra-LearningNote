# 推理性能评估

## 1. evalscope

evalscope是一个用于模型评测的工具，其可以指定网络上的模型或者本地的模型进行测试，在进行模型部署前最好先安装vllm来保证评测模型正常部署

```shell
# 安装额外依赖
pip install evalscope[perf] -U
```

评测脚本

```shell
evalscope perf \
 --model 'Qwen/Qwen2.5-0.5B-Instruct' \
 --attn-implementation flash_attention_2 \
 --number 20 \
 --parallel 2 \
 --port 9998\
 --api local \
 --dataset openqa

```