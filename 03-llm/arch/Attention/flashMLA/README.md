# FlashMLA

FlashMLA是deepseek开发的一个mla的高性能实现

其分为dense和sparse两部分

## 1. dense

### 1.1 共享内存配置

dense的mla其实计算的形式其实和普通MHA没有差异，差异的点在于 $d_{qk}$ 和 $d_v$ 是不同的，qk会多出一个rope的维度，一个比较经典的配置是 $d_{qk} = 196$, $d_v = 128$。

在MLA上面的配置下，share memory的占用来自下面几个部分

- Q: 128x182x2stage = 96KB，这个2stage是给2-CTA用的
- K: 128x192x1 = 48KB
- V/O1: 128x128x1stage = 32KB： O的部分输出
- O0: 128x128 = 32KB： O的另外一半的输出
- 合计209KB，卡着227KB上限

对于128的普通attention而言

- Q：128x128x2stage = 64KB，这个2stage是给2-CTA使用
- K/V共享环形缓冲：128x128x3（stage）=96KB
- O: 128x128x2 stage = 64KB

合计224KB，卡着的227KB的上限

fa4在192/128(bf16,1-CTA, q_stage=2, tile = 256x128)的smem配置如下所示

- Q(含O别名)： Q=2x128x182x2B = 96KB，O = 2x128x128x2B=64B，但overlap_sO_sQ=True，所以O直接叠在Q的前64KB上
- KV非均匀换，采用大小格子交错的形式，大格48KB+小格32KB+大格48KB， 3级
合计224KB

### 1.2 并行方式

## 2. sparse
