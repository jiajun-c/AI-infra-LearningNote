# BPE 算法


BPE算法是一种基于统计的子词分词算法，其核心思想是通过迭代合并高频字符对生成子词单元，从而平衡词表大小和未登录词问题

算法步骤
- 初始化词汇表，如 `new` $\to$ `n, e, w, </w>`，频率为3
- 统计字符对频率,`(n, e):5`, `(e, w):5`
- 合并最高频率，例如 `(n, e)`$\to$ ne
- 迭代直到目标词表的大小

如下是一个BPE算法的例子，每次选出频率最高的字符对，并合并，直到词表大小达到目标大小

```python3
    def apply_bpe(self, byte_seq: List[bytes]) -> List[bytes]:
        word = tuple(byte_seq)
        pairs = Tokenizer.get_pairs(word)
        if not pairs:
            return list(word)

        while True:
            # pick the highest‑priority merge that exists in this word
            min_pair = None
            min_rank = None
            for pair in pairs:
                rank = self.bpe_ranks.get(pair)
                if rank is not None and (min_rank is None or rank < min_rank):
                    min_pair, min_rank = pair, rank
            if min_pair is None:
                break

            first, second = min_pair
            new_word = []
            i = 0
            while i < len(word):
                # if first+second occurs here, merge it
                if i < len(word)-1 and word[i] == first and word[i+1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            word = tuple(new_word)
            pairs = Tokenizer.get_pairs(word)

        return list(word)
```