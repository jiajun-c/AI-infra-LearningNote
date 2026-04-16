#!/bin/bash

# SM 放置策略性能对比快速测试脚本

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================================"
echo "SM 放置策略性能对比测试"
echo "========================================================"

# 编译
echo "正在编译..."
nvcc -o main main.cu -O3 -arch=sm_90 -diag-suppress 177

if [ $? -ne 0 ]; then
    echo "编译失败!"
    exit 1
fi

echo "编译完成，开始测试..."
echo ""

# 运行测试
./main

echo ""
echo "========================================================"
echo "测试完成! 结果已保存到 performance_summary.md"
echo "========================================================"
