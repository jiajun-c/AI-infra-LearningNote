# 梯度累加步数

假设我们希望模型使用的batch大小为256,但是GPU显存下只能放得下64个样本和中间的激活值，那么这个时候可以使用梯度累加步数，将实际喂给 GPU 的批次大小（称为 Micro-Batch Size）设为 64，并将 gradient_accumulation_steps 设为 4。这样，GPU 分 4 次计算，每次计算 64 个样本的梯度并累加起来，等攒够了 4 次后，再统一更新一次模型权重。

对应的伪代码

```shell
# 初始化累加步数
accumulation_steps = 4  
optimizer.zero_grad() # 训练开始前清空梯度

for i, (inputs, labels) in enumerate(dataloader):
    # 1. 前向传播
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    
    # 2. 缩放 Loss：因为梯度是累加的，为了保证梯度的量级等效于大 Batch，需要除以累加步数
    loss = loss / accumulation_steps
    
    # 3. 反向传播：计算梯度并累加到原有的梯度上（此时不更新权重）
    loss.backward()
    
    # 4. 判断是否达到了累加步数
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()      # 更新模型权重
        optimizer.zero_grad() # 清空梯度，准备下一轮累加
```