# Torch graph

在cuda中支持cuda graph模式从而来消除CPU侧多个kernel启动的开销，在torch中也支持了这一模型

torch的cuda图模式主要分为两个部分

- 分析和定义桶，将常见的张量形状等定义
- 进行warmup，使得上下文分配等达到稳定，为每个桶创建专属的cuda graph
- 允许时动态选择和重放

如下所示是一个torch cuda 图的例子

```python
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()
```