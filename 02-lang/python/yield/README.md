# yield

python中提供了生成器的语法，其不同于迭代器，其具有惰性加载的特点，可以让内存中始终只保存一份数据

## 例子：惰性读取超大日志

```python
# 核心业务代码：使用 yield 按需逐行读取，内存占用几乎为 0
def extract_errors_from_huge_log(file_path):
    """
    一个生成器函数，专门用来处理超大日志文件。
    每次只在内存中保留一行数据。
    """
    print("生成器启动，准备开始扫描日志...")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f, start=1):
            if "ERROR" in line:
                # 关键点：遇到 ERROR，就用 yield 把这一行和行号抛出去，然后函数在此处暂停！
                yield {"line_number": line_number, "content": line.strip()}
    print("日志扫描完毕！")


# ----- 实际调用的地方 -----
print("开始任务...")

# 1. 这里只是创建了生成器对象，文件并没有被真正读取，连 1KB 内存都没消耗
error_scanner = extract_errors_from_huge_log("server_error.log") 

# 2. 用 for 循环驱动生成器。每次循环，生成器往下执行一步，遇到 yield 就暂停。
for error_item in error_scanner:
    print(f"在第 {error_item['line_number']} 行发现错误: {error_item['content']}")
    
    # 因为是按需生成的，你甚至可以随时终止它，前面的工作完全没有浪费
    if error_item['line_number'] > 1000:
         print("已经找到足够的错误，停止扫描。")
         break
```

## 生成器是双向通道

`yield` 不是单向的「抛数据出去」，它有两个方向：

```python
def g():
    x = yield 1     # 1 流向调用方；x 是从调用方 send() 进来的值
    return 42       # 42 不是送给消费者的
```

- **`yield v`**：`v` 流向调用 `next()` / `send()` 的一方；而 `send(x)` 传进来的 `x` 成为 `yield` 表达式的值
- **`return v`**：不产生任何 yield，而是让生成器抛 `StopIteration(v)`，值挂在异常的 `.value` 上

所以 `next()` 拿到的只是 yield 出来的值；生成器的**返回值只能从 `StopIteration.value` 里取**：

```python
gen = g()
next(gen)          # 1        ← yield 出来的
try:
    next(gen)
except StopIteration as e:
    e.value        # 42       ← return 出来的
```

这也解释了 `for` 为什么能正常工作：`for` 内部就是不断 `next()`，把 `StopIteration` 当作「循环正常结束」吞掉。

## 为什么不能 raise StopIteration

PEP 479（Python 3.7 起）规定：生成器内部冒出的 `StopIteration` 会被**转换成 `RuntimeError`**。

因为 `StopIteration` 被 `for` 和 `yield from` 当作「迭代结束」的信号。允许在生成器里手抛它，就把「函数正常返回」和「迭代器自然耗尽」两件事搅在一起，会静默地出错。所以唯一的合法出口是 `return v`。

## yield from

`yield from iterable` 把内层迭代器的 yield 全部透传给外层，并把**内层的 `return` 值**当作自己这个表达式的值：

```python
result = yield from inner   # result 就是 inner 的 return 值
```

等价于（省掉异常转发的简化版）：

```python
_i = iter(inner)
_y = None
while True:
    try:
        _sent = yield _y
    except StopIteration:
        return
    else:
        try:
            _y = next(_i) if _sent is None else _i.send(_sent)
        except StopIteration as e:
            return e.value       # ★ 内层 return 值在这里被接住
```

`asyncio` 就是这么实现的：`await x` ≡ `yield from x.__await__()`。**async/await 只是语法糖，底层机制和 yield from 完全相同。**

手写迷你 asyncio 见 [lab5_mini_asyncio.py](lab5_mini_asyncio.py)。

## 例子

在库代码中经常有with语句，假设我们希望可以监控一下内部的操作前后的变化，可以yield前后加上监控的代码，然后打印前后的时间差或者变化。

```python
from contextlib import contextmanager
import time
import torch

@contextmanager
def timer(name):
    t = time.perf_counter()
    yield
    print(f"{name}: {time.perf_counter() - t:.3f}s")

with timer("forward"):
    time.sleep(1)
    a = torch.randn(1024, 1024)
    b = torch.randn(1024, 1024)
    c = a @ b
```