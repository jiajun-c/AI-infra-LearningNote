import time
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)


class SyntheticDataset(Dataset):
    def __init__(self, size=10000, feature_dim=224, transform_delay=0.001):              
        self.size = size
        self.feature_dim = feature_dim
        self.transform_delay = transform_delay

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        data = torch.randn(3, self.feature_dim, self.feature_dim)                        
        label = torch.randint(0, 10, (1,)).item()                                        
        if self.transform_delay > 0:                                                     
            time.sleep(self.transform_delay)                                             
        return data, label

    def __getitems__(self, indices):                                                                                                                             
        n = len(indices)                                                               
          # 向量化生成：一次调用代替 N 次单独调用                                          
        data = torch.randn(n, 3, self.feature_dim, self.feature_dim)                     
        labels = torch.randint(0, 10, (n,))                                              
          # 模拟批次级 I/O：整批只 sleep 一次，                                            
          # 而不是每个样本 sleep 一次（例如一次 SQL 查询取 N 行）                          
        if self.transform_delay > 0:
            time.sleep(self.transform_delay)                                             
        return [(data[i], labels[i].item()) for i in range(n)]


class SmallTransformerModel(nn.Module):
    def __init__(self):                                                                  
        super().__init__()                                                               
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=4, padding=3),                        
            nn.ReLU(),                                                                   
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),                       
            nn.ReLU(),                                                                   
            nn.AdaptiveAvgPool2d((7, 7)),                                                
        )                                                                                
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128, batch_first=True                   
        )                                                                                
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)            
        self.classifier = nn.Linear(64, 10)                                              
                                                                                           
    def forward(self, x):                                                                
        x = self.features(x)  # (B, 64, 7, 7)                                            
        B, C, H, W = x.shape                                                             
        x = x.view(B, C, H * W).permute(0, 2, 1)  # (B, 49, 64)
        x = self.transformer(x)  # (B, 49, 64)                                           
        x = x.mean(dim=1)  # (B, 64)                                                     
        return self.classifier(x)   

def create_model():
    """创建一个 conv+transformer 模型用于基准测试。"""
    return SmallTransformerModel().to(device)

benchmark_dataset = SyntheticDataset(size=512, feature_dim=224, transform_delay=0.005)

class DataPrefetcher:
    """
    GPU-side prefetcher: 在独立 CUDA stream 上把下一批数据 H2D 进来,
    让数据搬运和当前 batch 的 GPU 计算重叠, 减少 GPU 空转。

    用法:
        loader = DataLoader(ds, batch_size=64, num_workers=2)
        for data, labels in DataPrefetcher(loader, device):
            output = model(data)         # data/labels 已经在 device 上
            loss = criterion(output, labels)
            ...

    关键点:
        - H2D 在 self.stream (独立 stream) 上做, 不阻塞默认 stream
        - 主 stream (forward 用的那个) 用 wait_stream 同步
        - 不依赖 Apex, 不依赖 torchdata, 纯 PyTorch 1.9+ 都能跑
    """

    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        # 关键: 独立 CUDA stream, H2D 在它上面跑, 不阻塞 forward 的默认 stream
        self.stream = (
            torch.cuda.Stream(device=device) if device.type == "cuda" else None
        )
        self.loader_iter = None
        self.next_data = None
        self.next_labels = None

    def preload(self):
        """在 self.stream 上把下一批搬到 device, 主线程继续算上一批。"""
        try:
            self.next_data, self.next_labels = next(self.loader_iter)
        except StopIteration:
            self.next_data = self.next_labels = None
            return

        if self.stream is not None:
            # 用 self.stream 而不是默认 stream, 这样 H2D 不阻塞 forward
            with torch.cuda.stream(self.stream):
                self.next_data = self.next_data.to(self.device, non_blocking=True)
                self.next_labels = self.next_labels.to(self.device, non_blocking=True)
        else:
            # CPU device, 直接搬
            self.next_data = self.next_data.to(self.device)
            self.next_labels = self.next_labels.to(self.device)

    def __iter__(self):
        self.loader_iter = iter(self.loader)
        self.preload()
        return self

    def __next__(self):
        if self.next_data is None:
            raise StopIteration
        # 等 self.stream 上的 H2D 完成
        if self.stream is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
        data, labels = self.next_data, self.next_labels
        # 趁主线程在算当前 batch, 后台开始搬下一批
        self.preload()
        return data, labels 


def train_and_benchmark(loader, max_batches=160, epochs=10, prefetch_device=None):                                                                                  
    model = create_model()                                                               
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)                             
    criterion = nn.CrossEntropyLoss()                                                    
                  
    start_time = time.perf_counter()                                                     
    total_loss = 0.0
    num_batches = 0                                                                      
                  
    for epoch in range(epochs):                                                          
        if prefetch_device is not None:
            data_iter = DataPrefetcher(loader, prefetch_device)                          
        else:
            data_iter = loader                                                           
                  
        for data, labels in data_iter:                                                   
            if prefetch_device is None:
                data = data.to(device, non_blocking=True)                                
                labels = labels.to(device, non_blocking=True)                            
                                                                                           
            output = model(data)                                                         
            loss = criterion(output, labels)                                             
                                                                                           
            optimizer.zero_grad()
            loss.backward()                                                              
            optimizer.step()                                                             
   
            total_loss += loss.item()                                                    
            num_batches += 1
                                                                                           
            if num_batches >= max_batches:
                break                                                                    
        if num_batches >= max_batches:
            break

    if torch.cuda.is_available():                                                        
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time                                           
                                                                                           
    return elapsed, total_loss / num_batches

# baseline_loader = DataLoader(
#       benchmark_dataset,
#       batch_size=32,
#       shuffle=True,                                                                        
#       num_workers=0,
#       pin_memory=False,                                                                    
#   )               
                                                                                           
# print("\n=== Progressive Optimization Results ===")
# print("\nBaseline (num_workers=0, pin_memory=False):")
# baseline_time, baseline_loss = train_and_benchmark(baseline_loader)
# print(f"  Time: {baseline_time:.4f}s | Loss: {baseline_loss:.4f}")                       
# prev_time = baseline_time

batch_dataset = SyntheticDataset(size=1000, transform_delay=0)                           
                                                                                           
def benchmark_batch_size(batch_size, num_batches=10):                                    
    """使用特定 batch size 对数据加载进行基准测试。"""                                   
    loader = DataLoader(batch_dataset, batch_size=batch_size, shuffle=True)              
    start = time.perf_counter()
    for i, (data, labels) in enumerate(loader):                                          
        if i >= num_batches:                                                             
            break                                                                        
        data = data.to(device, non_blocking=True)                                        
        _ = data.sum()                                                                   
    if torch.cuda.is_available():                                                        
        torch.cuda.synchronize()                                                         
    elapsed = time.perf_counter() - start                                                
    return elapsed                                                                       
                                                                                           
  #基准测试不同的 batch size                                                               
# print("\nBatch size comparison (isolated benchmark):")
# for bs in [16, 32, 64, 128]:                                                             
#     elapsed = benchmark_batch_size(bs)
#     print(f"  Batch size {bs:3d}: {elapsed:.4f}s for 10 batches")       

workers_loader = DataLoader(                                                             
    benchmark_dataset,
    batch_size=32,                                                                       
    shuffle=True,
    num_workers=4,
    prefetch_factor=2,
    pin_memory=False,
)

baseline_loader = DataLoader(                                                             
    benchmark_dataset,
    batch_size=32,                                                                       
    shuffle=True,
    num_workers=1,
    prefetch_factor=2,
    pin_memory=False,
)

baseline_time, workers_loss = train_and_benchmark(baseline_loader)

print("\n+ num_workers=4, prefetch_factor=2:")
workers_time, workers_loss = train_and_benchmark(workers_loader)
print(f"  Time: {workers_time:.4f}s | Loss: {workers_loss:.4f}")
print(f"  Speedup vs baseline: {baseline_time / workers_time:.2f}")


# ============================================================
# DataLoader 加速比扫描: 三个独立视角
# ============================================================
# DataLoader 性能主要受两个旋钮影响:
#   - num_workers    : 控制 __getitem__ 的进程并行度
#   - prefetch_factor: 每个 worker 提前准备多少 batch
#
# 这两个参数互相耦合, 但**只看一个变量**时最容易读懂边际收益。
# 所以拆成 3 个独立表:
#
#   表 1: 完整 2D 矩阵 (哪个组合最快)
#   表 2: 固定 prefetch_factor, 扫 num_workers (加 worker 的收益)
#   表 3: 固定 num_workers, 扫 prefetch_factor (加 prefetch 的收益)

WORKER_COUNTS = [0, 1, 2, 4, 8]
PREFETCH_FACTORS = [1, 2, 4, 8]


def make_loader(num_workers, prefetch_factor):
    """构造一个 DataLoader。

    关键防御:
      - num_workers=0 时 prefetch_factor 必须传 None (PyTorch 强制)
      - num_workers>0 时显式 multiprocessing_context='fork',
        避免某些平台默认 spawn 导致 worker 重新 import 主模块, 引发死锁
      - 加 timeout, worker 卡死时 DataLoader 抛错而不是挂死主进程
    """
    kwargs = dict(
        dataset=benchmark_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    if num_workers == 0:
        # prefetch_factor 不生效, 也不允许传非 None
        return DataLoader(**kwargs)
    kwargs["prefetch_factor"] = prefetch_factor
    # 显式 fork, 避免 PyTorch 在某些平台退回 spawn
    kwargs["multiprocessing_context"] = mp.get_context("fork")
    # worker 卡死时直接报错, 而不是无限等下去
    kwargs["timeout"] = 60  # 秒
    return DataLoader(**kwargs)


def safe_benchmark(num_workers, prefetch_factor):
    """跑一次 benchmark, 出错返回 inf/nan, 不让整个扫描挂掉。"""
    loader = make_loader(num_workers, prefetch_factor)
    try:
        return train_and_benchmark(loader)
    except RuntimeError as e:
        print(f"  [WARN] nw={num_workers}, pf={prefetch_factor} failed: {e}")
        return float("inf"), float("nan")


# 一次跑完所有组合, 后面的三个表都从这里读
results = {}  # (nw, pf) -> (time, loss)
for nw in WORKER_COUNTS:
    for pf in PREFETCH_FACTORS:
        t, loss = safe_benchmark(nw, pf)
        results[(nw, pf)] = (t, loss)


# ----------------------------------------------------------------
# 表 1: 完整 2D 矩阵 (哪个组合最快)
# ----------------------------------------------------------------
print("\n=== 表 1: 完整 2D 加速比矩阵 (vs 最慢的那一格) ===")

# baseline = 最慢那一格, 一般是 (num_workers=0, pf=1)
baseline_key = max(results, key=lambda k: results[k][0])
baseline_t = results[baseline_key][0]
print(f"基准 (最慢): nw={baseline_key[0]}, pf={baseline_key[1]}, "
      f"time={baseline_t:.4f}s\n")

row_label = "num_workers\\prefetch"
print(f"{row_label:<22} | " +
      " | ".join(f"pf={p:<3}" for p in PREFETCH_FACTORS))
print("-" * (22 + 11 * len(PREFETCH_FACTORS)))
for nw in WORKER_COUNTS:
    cells = []
    for pf in PREFETCH_FACTORS:
        t = results[(nw, pf)][0]
        speedup = baseline_t / t if t != float("inf") else 0
        cells.append(f"{speedup:5.2f}x")
    print(f"  num_workers={nw:<3}            | " + " | ".join(cells))
print()


# ----------------------------------------------------------------
# 表 2: 固定 prefetch_factor=2, 扫 num_workers
# ----------------------------------------------------------------
print("=== 表 2: 固定 prefetch_factor=2, 扫 num_workers ===")
print("回答: 加 worker 的边际收益是多少? 边际在哪个 nw 开始递减?\n")

PF_FIXED = 2
base_t_pf = results[(1, PF_FIXED)][0]  # 基线: nw=1, pf=PF_FIXED
print(f"基线: num_workers=1, prefetch_factor={PF_FIXED}, "
      f"time={base_t_pf:.4f}s\n")

print(f"{'num_workers':<14} | {'time (s)':<10} | {'speedup':<10} | "
      f"{'marginal':<10}")
print("-" * 56)
prev_speedup = 1.0
for nw in WORKER_COUNTS:
    t = results[(nw, PF_FIXED)][0]
    speedup = base_t_pf / t if t != float("inf") else 0
    marginal = speedup / prev_speedup if prev_speedup > 0 else 0
    print(f"  {nw:<12} | {t:.4f}     | {speedup:.2f}x      | "
          f"{marginal:.2f}x")
    prev_speedup = speedup if speedup > 0 else prev_speedup
print()


# ----------------------------------------------------------------
# 表 3: 固定 num_workers=4, 扫 prefetch_factor
# ----------------------------------------------------------------
print("=== 表 3: 固定 num_workers=4, 扫 prefetch_factor ===")
print("回答: prefetch_factor 调到几最划算? 边际递减点在哪?\n")

NW_FIXED = 4
base_t_nw = results[(NW_FIXED, 1)][0]  # 基线: nw=NW_FIXED, pf=1
print(f"基线: num_workers={NW_FIXED}, prefetch_factor=1, "
      f"time={base_t_nw:.4f}s\n")

print(f"{'prefetch_factor':<14} | {'time (s)':<10} | {'speedup':<10} | "
      f"{'marginal':<10}")
print("-" * 56)
prev_speedup = 1.0
for pf in PREFETCH_FACTORS:
    t = results[(NW_FIXED, pf)][0]
    speedup = base_t_nw / t if t != float("inf") else 0
    marginal = speedup / prev_speedup if prev_speedup > 0 else 0
    print(f"  {pf:<12} | {t:.4f}     | {speedup:.2f}x      | "
          f"{marginal:.2f}x")
    prev_speedup = speedup if speedup > 0 else prev_speedup
print()


# ----------------------------------------------------------------
# 总结: 最佳配置
# ----------------------------------------------------------------
valid_results = {k: v for k, v in results.items() if v[0] != float("inf")}
if valid_results:
    best_key = min(valid_results, key=lambda k: valid_results[k][0])
    print(f"最佳配置: num_workers={best_key[0]}, prefetch_factor={best_key[1]}")
    print(f"  加速比 vs baseline: {baseline_t / valid_results[best_key][0]:.2f}x")
    print(f"  时间: {valid_results[best_key][0]:.4f}s")
else:
    print("[WARN] 所有组合都失败了, 没法给出最佳配置")