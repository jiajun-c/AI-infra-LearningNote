# CP 并行

Context Parallelism（CP）是序列并行的一种方式，将长序列切分到多个 GPU 上，解决单卡显存放不下完整序列的问题。三种主流方案从不同维度切分：

|特性|Megatron-SP|Ring Attention|Ulysses|
|---|---|---|---|
|切分维度|序列维度 (token)|序列维度 (token)|**Head 维度**|
|QKV 投影|需要 all-gather|本地计算|**本地计算，无通信**|
|Attention 计算|本地计算|P2P 环传 KV，循环 N 步|**all-to-all 后本地算**|
|通信模式|all-gather / reduce-scatter|P2P send/recv 环|all-to-all (2 次)|
|通信量 (每层)|$4sh$|$2sh$ (ring)|$6sh/P$ per GPU|
|优势|实现简单|通信量低，可 overlap|QKV/Output 投影零通信|
|适用场景|中等序列长度|极长序列 (如 1M)|多 head 模型 (如 GQA 少 KV head 时更优)|

## 1. Megatron-SP

Megatron-LM 的序列并行按 token 维度切分：每张卡持有 $s/P$ 个 token 的完整 hidden state。per-token 操作（LayerNorm、Dropout）零通信；线性层前 all-gather 恢复完整序列，计算后再 reduce-scatter 分回，通信量 $4sh$ per layer。详见 [Megatron-SP 切分公式](./Megtron-SP/README.md)。

## 2. Ring Attention

在序列维度切分，每张卡只持有 $Q$ 的一段和 $K$、$V$ 的一段，通过 P2P 环传递 $K$、$V$ 完成全局 attention，详见 [Ring Attention](./ringAttention/README.md)。

## 3. Ulysses

DeepSpeed 提出，在 **head 维度切分**：QKV 投影和 Output 投影时无需任何通信；通过 2 次 all-to-all 在 head 并行和序列并行之间切换，详见 [Ulysses 切分公式](./ulysses/README.md)。
