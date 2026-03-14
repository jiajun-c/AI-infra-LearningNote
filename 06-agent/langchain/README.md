# langchain 入门

## 1. LCEL表达式

lcel是用于langchain的表达式语法，其可以轻松地将链条组合在一起

如下所示，将prompt，model和输出解析变为一个链条，其思想类似linux shell的管道符 

```python
chain = prompt | model | output_parser
```

异步输出

```python3
async for s in chain.astream({"topic": "bears"}):
    print(s.content, end="", flush=True)  
```

批量输出

```python3
# 执行批量请求
results = chain.batch(inputs, config={"max_concurrency": 5})

# 5. 逐条打印完整结果（非流式）
for i, result in enumerate(results, 1):
    print(f"{result}\n")
```
