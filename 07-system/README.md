# 系统与设备基础

先看 [CPU Cache](./cpu/x86/cache/README.md) 和 [系统内存](./memory/README.md)，再读 [NUMA](./numa/README.md)、[操作系统 I/O](./os/io_uring/README.md) 与 [网络](./net/README.md)。设备专题可从 [GPU 拓扑](./gpu/topo/README.md)、[NPU](./npu/README.md) 或 [设备信息](./device/README.md) 进入。

## XPU 架构概述

XPU 是异构计算加速器的统称，包括 GPU、NPU、FPGA 等多种计算架构。

## 1. XPU 分类

### 1.1 GPU (图形处理器)

- NVIDIA GPU (CUDA 架构)
- AMD GPU (ROCm 架构)
- Intel GPU (OneAPI/Xe 架构)

### 1.2 NPU (神经网络处理器)

- 华为昇腾
- 谷歌 TPU
- 寒武纪

### 1.3 FPGA (现场可编程门阵列)

## 2. 架构对比

| 架构 | 特点 | 适用场景 |
|------|------|----------|
| GPU | 高并行，通用性强 | 图形渲染、深度学习 |
| NPU | 专用 AI 加速 | 推理、训练 |
| FPGA | 高度可定制 | 专用算法加速 |

## 3. 编程模型

### 3.1 CUDA (NVIDIA)

### 3.2 ROCm (AMD)

### 3.3 OneAPI (Intel)

## 4. 发展趋势
