"""
yield Lab 4：throw() 和 close() —— 异常注入与清理
===================================================
目标：理解生成器的异常处理机制：
  - throw(exc)：在 yield 处注入异常，生成器可以 try/except 捕获并继续
  - close()：注入 GeneratorExit，生成器必须 return 或再次抛出，用于资源清理
  - finally 块：不管是正常结束还是被 close()，finally 都会执行

任务：
  1. 补全 resilient_gen：能捕获 ValueError 并继续运行，捕获 RuntimeError 则停止
  2. 补全 resource_gen：模拟打开/关闭资源，在 finally 里做清理
  3. 观察 close() 和直接让生成器耗尽的区别

问题思考：
  - GeneratorExit 是 BaseException 还是 Exception？为什么不能被 except Exception 捕获？
  - asyncio 的 Task.cancel() 底层就是向协程 throw(CancelledError)，和这里有何关联？
"""

class MyError(Exception):
    pass

def resilient_gen():
    """
    能承受 ValueError，遇到 RuntimeError 则停止。

    TODO:
      i = 0
      while True:
          try:
              yield i
              i += 1
          except ValueError as e:
              print(f"  捕获 ValueError: {e}，继续运行")
          except RuntimeError as e:
              print(f"  捕获 RuntimeError: {e}，停止")
              return
    """
    pass

def resource_gen(name: str):
    """
    模拟需要资源管理的生成器（文件、连接等）。
    无论正常结束还是被 close()，finally 都会执行。

    TODO:
      print(f"  [{name}] 打开资源")
      try:
          for i in range(5):
              yield i
      except GeneratorExit:
          print(f"  [{name}] 收到 GeneratorExit")
          # 不能 yield，只能 return 或让异常传播
          return
      finally:
          print(f"  [{name}] 关闭资源（finally）")
    """
    pass

def main():
    print("── throw() 注入异常 ──")
    gen = resilient_gen()
    next(gen)
    # TODO:
    #   print(gen.send(None))     # 推进到 yield 1
    #   gen.throw(ValueError, "可恢复的错误")
    #   print(next(gen))          # 继续运行，yield 1（i 没有递增）
    #   gen.throw(RuntimeError, "致命错误")  # 生成器停止

    print("\n── close() 触发清理 ──")
    gen = resource_gen("A")
    print(f"  next → {next(gen)}")
    print(f"  next → {next(gen)}")
    gen.close()   # 注入 GeneratorExit，触发 finally
    print("  close() 返回后，生成器已终止")

    print("\n── 正常耗尽也会触发 finally ──")
    gen = resource_gen("B")
    # TODO: 用 for 遍历直到结束，观察 finally 何时触发

    print("\n── GeneratorExit 是 BaseException ──")
    import traceback
    def bad_gen():
        try:
            yield 1
        except Exception:
            print("  Exception 捕获不到 GeneratorExit")
            yield 2   # 如果在 GeneratorExit 后 yield，抛 RuntimeError
    gen = bad_gen()
    next(gen)
    try:
        gen.close()
    except RuntimeError as e:
        print(f"  RuntimeError: {e}")

if __name__ == "__main__":
    main()
