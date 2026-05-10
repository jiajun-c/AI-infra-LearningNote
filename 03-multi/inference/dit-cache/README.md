# Dit Cache

Dit的Cache策略指的是由于在相邻的时间步之间，Attention和MLP层的输出可能高度类似，所以可以缓存对应的结果。每隔cache_threshold步才会重新计算一下Attention和MLP的输出

## 1. FORA

早期的工作如FORA，是第一个将UNet部分的caching策略用到了Dit中

核心逻辑如下，只在 step 被 cache_threshold整除的step进行caching

```python
    def forward(self, x, c, layer, step, cache=None, cache_attn=None, cache_mlp=None, numstep=250, cache_threshold=0.95):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        sim_threshold = 0.95
        cache_type = cache['save_cache']
        cache_subtype = cache['cache_subtype']

        if cache_type == 'boost_infer_static':
            if (step == int(numstep-1)) or (step % int(cache_threshold) == 0):
                should_cache = True
            else:
                should_cache = False

            if should_cache:
                cache[-1][layer]['attn'] = self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
                x = x + gate_msa.unsqueeze(1) * cache[-1][layer]['attn']
                cache[-1][layer]['mlp'] = self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
                x = x + gate_mlp.unsqueeze(1) * cache[-1][layer]['mlp']
            else:
                x = x + gate_msa.unsqueeze(1) * cache[-1][layer]['attn']
                x = x + gate_mlp.unsqueeze(1) * cache[-1][layer]['mlp']

        elif cache_type == 'save_cache':
            cache[step][layer]['attn'] = self.attn(modulate(self.norm1(x), shift_msa, scale_msa)).detach().cpu()
            x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
            cache[step][layer]['mlp'] = self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp)).detach().cpu()
            x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))

        else:
            x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
            x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))

        return x
```


## 2. 