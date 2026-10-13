# 交叉熵损失函数

$L_{CE} = -\sum_{i=1}^n y_i \log(\hat{y_i})$

对于大语言模型中的交叉熵损失，通常写成

$L = -\frac{1}{T}\sum_{i=1}^T \log P_{\theta}(x_{t+1}|x_{<=t})$

- B：batch size
- S：seqlen
- V：vocabulary size

LLM的最后一层通过LM head产生logits[B,S,V]，表示对不同输出token的打分，然后使用softmax将logits转换为概率分布，所有token的概率之和为1

由于one hot标签中只有正确token的位置为1,其他位置都是0

所以$L_t = -\log p_y$ 

合并softmax和cross-entropy可以得到

$L_t = \log(\sum_{j=0}^{V-1}e^{z_j}) - z_y$
