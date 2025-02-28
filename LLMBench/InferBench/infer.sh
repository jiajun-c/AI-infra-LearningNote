evalscope perf \
 --model 'Qwen/Qwen2.5-0.5B-Instruct' \
 --attn-implementation flash_attention_2 \
 --number 20 \
 --parallel 2 \
 --api local \
 --dataset openqa
