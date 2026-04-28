# yield

python中提供了生成器的语法，其不同于迭代器，其具有惰性加载的特点，可以让内存中始终只保存一份数据

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