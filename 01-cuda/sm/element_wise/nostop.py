#!/usr/bin/env python3
"""
nostop.py - 保持 GPU 卡 0 每隔 600 秒高负载一次算矩阵乘

用于防止 GPU 进入低功耗状态或保持 GPU 活跃。
"""

import os
import time
import torch

# 设置只使用 GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

MATRIX_SIZE = 8192  # 矩阵大小，8192x8192 足够产生高负载
INTERVAL = 600  # 间隔时间（秒）

def matmul_benchmark():
    """执行矩阵乘法"""
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 开始矩阵乘法计算...")

    # 创建随机矩阵
    A = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device='cuda', dtype=torch.float32)
    B = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device='cuda', dtype=torch.float32)

    # 执行矩阵乘法
    C = torch.matmul(A, B)

    # 同步确保计算完成
    torch.cuda.synchronize()

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 矩阵乘法完成 ({MATRIX_SIZE}x{MATRIX_SIZE})")

def main():
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] nostop.py 启动")
    print(f"配置：GPU 0, 矩阵大小 {MATRIX_SIZE}x{MATRIX_SIZE}, 间隔 {INTERVAL}秒")
    print(f"按 Ctrl+C 停止\n")

    # 立即执行一次
    matmul_benchmark()

    while True:
        print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] 等待 {INTERVAL}秒后下一次计算...")
        time.sleep(INTERVAL)
        matmul_benchmark()

if __name__ == "__main__":
    main()
