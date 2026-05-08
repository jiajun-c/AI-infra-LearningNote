# LDM(latent Diffusion Model)

LDM 先使用VAE把图像压缩到低维的latent，相比于ADM而言，扩散过程在latent space，计算成本更低，用 Cross-Attention 替代 Classifier Guidance，直接以文本/图像为条件（不需要单独训分类器）
