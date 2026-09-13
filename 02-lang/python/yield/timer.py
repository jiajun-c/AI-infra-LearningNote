from contextlib import contextmanager
import time
import torch

@contextmanager
def timer(name):
    t = time.perf_counter()
    yield
    print(f"{name}: {time.perf_counter() - t:.3f}s")

def func():
    time.sleep(1)
    a = torch.randn(1024, 1024)
    b = torch.randn(1024, 1024)
    yield
    time.sleep(1)
    c = a @ b
    return c

with timer("forward"):
    for _ in func():
        pass