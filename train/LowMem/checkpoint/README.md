# 检查点机制

在传统反向传播中，需要保存各层的激活值，从而将其用于反向传播。但是这样会导致显存的占用较大。如果全部都进行重新的计算，那么将会导致计算时间大幅增加，检查点机制则是选择了一个折中的方案，保留部分的激活值，从而减少显存占用，当需要的前向传播结果不在检查点内时，则从最近的检查点开始进行重新计算。

在hf的代码中，仅需要将gradient_checkpointing设置为True即可。

```python3
from transformers import TrainingArguments

args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    fp16=True.
)
```