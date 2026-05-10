# text2X

生成领域通常用t2x去表示从文本到的若干多模态的输出
- t2i：text to image
- t2v: text to video

## t2i

像 ViT 一样，把一张静态图片切成比如 $16 \times 16$ 的小网格（Patches），每个 Patch 展平后变成一个 Token。序列长度 $N = H \times W$。

## t2v

视频是由多帧组成的。文生视频不仅要在空间上切，还要在时间上切。Sora 等模型采用的是 时空管（Tubelet） 结构，把连续的几帧中的同一块区域打包成一个三维的 Patch。序列长度爆炸式增长，变为 $N = T \times H \times W$。