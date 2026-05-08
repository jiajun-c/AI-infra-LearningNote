# DIT 

DiT 的核心思想：用 Transformer 替换扩散模型中传统的 U-Net 骨干网络，在潜空间（latent space）上操作。

```shell
输入图像潜变量 (N, 4, H, W)
       ↓ PatchEmbed + 位置编码
图像 tokens (N, T, D)     ← T = H*W / patch_size²
       ↓
    ┌──────────────────────────────────┐
    │  条件信号 c = t_emb + y_emb      │  ← 时间步 t + 类别标签 y
    └──────────────────────────────────┘
       ↓ × depth 个 DiTBlock
       ↓ FinalLayer
预测噪声 (N, out_channels, H, W)
```

## 1. 输入嵌入

- 图片token：用 PatchEmbed（本质是 Conv2d）把潜变量切成 patch，展平为 token 序列
- 时间步嵌入：先做 sinusoidal 频率编码（同 Transformer 位置编码的思路）
- 类别标签嵌入：简单 Embedding 表查表 → y_emb，训练时有 10% 概率随机 drop（替换为特殊的"无条件"token），用于支持 classifier-free guidance

## 2. Dit Block


这是 DiT 最关键的设计。与标准 ViT Block 的区别在于 用条件信号调制 LayerNorm：


adaLN_modulation 把条件向量 c 映射为 6 个参数
```python
shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
    self.adaLN_modulation(c).chunk(6, dim=1)
```

modulate = x * (1 + scale) + shift  ← 自适应 LayerNorm
```python
x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
x = x + gate_mlp * self.mlp( modulate(self.norm2(x), shift_mlp, scale_mlp))
```
每个 block 对 Attention 和 MLP 分别有：

shift / scale：控制 LayerNorm 的均值和方差（让时间步/类别信息影响特征分布）
gate：控制残差连接的幅度（初始化为 0，使训练初期等效于恒等映射）
"Zero" 的含义：初始化时把 adaLN_modulation 最后一层的权重/偏置全设为 0 — models.py:208-210，保证训练起步稳定。

## 3. Final Layer

- 同样用 adaLN 调制（但只有 shift/scale，无 gate）
- 最后接一个线性层，把每个 token 从 D 维映射到 patch_size² × out_channels
- out_channels = 8（因为 learn_sigma=True，同时预测噪声均值和方差） unpatchify 把 token 序列重组回图像