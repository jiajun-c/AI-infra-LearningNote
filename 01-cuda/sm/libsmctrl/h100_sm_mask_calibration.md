# H100 SM 掩码映射校准总结

## 测试环境
- **GPU**: NVIDIA H100 80GB HBM3
- **SM 总数**: 132
- **CUDA**: 12.8
- **libsmctrl**: 基于 TMD/QMD hook 的 SM 掩码控制

## 关键发现

### 1. 掩码位和 SM ID 不是 1:1 映射

根据 libsmctrl README 的说明：

> Mask bit indexes do not directly correlate to software-visible TPC/SM IDs in V4 TMD/QMDs (Hopper+; compute capability 9.0). The mask bit indexes instead appear to correspond to on-chip-units, including disabled ones.

这意味着：
- 掩码位对应的是**物理 TPC 单元**（包括禁用的）
- 不是软件看到的连续 SM ID (0-131)
- H100 有 132 个 SM，但可能对应更多的物理 TPC 位置

### 2. 校准测试观察

从 `calibrate_sm_mask.cu` 的测试结果：

**单一掩码位测试**（每次只启用 1 个掩码位）：
```
掩码位 0 (0x0000000000000001) -> 启用了约 120 个 SM
掩码位 1 (0x0000000000000002) -> 启用了约 120 个 SM
...
```

**问题**：每次禁用单个掩码位后，仍然有超过 100 个 SM 被启用。
这说明：
1. 单个掩码位可能控制的不是一个 SM，而是一个 TPC 集群
2. H100 的 SM 到掩码位的映射是多对多的关系

### 3. GEMM 性能测试

| SM 限制 | TFLOPS | 相对性能 |
|--------|--------|----------|
| 132 | 21.10 | 100.0% |
| 96 | 15.11 | 71.6% |
| 64 | 15.39 | 72.9% |
| 48 | 21.17 | 100.3% ⚠️ |
| 32 | 18.68 | 88.5% |
| 24 | 7.34 | 34.8% |
| 16 | 4.22 | 20.0% |
| 8 | 1.57 | 7.5% |

**异常点**：48 SM 时性能反弹到 100%，说明掩码没有正确限制预期的 SM 数量。

### 4. H100 架构分析

H100 (Hopper 架构) 的特点：
- 132 个 SM
- 每个 GPC (Graphics Processing Cluster) 包含多个 TPC
- 每个 TPC 包含 2 个 SM

**可能的映射关系**：
```
物理布局（简化）:
GPC0: TPC0, TPC1, TPC2 -> SM0, SM1, SM2, SM3, SM4, SM5
GPC1: TPC3, TPC4, TPC5 -> SM6, SM7, SM8, SM9, SM10, SM11
...

掩码位可能对应的是 TPC ID，而不是 SM ID
```

## 结论

### libsmctrl 在 H100 上的局限性

1. **掩码映射复杂**: 掩码位不直接对应软件可见的 SM ID
2. **需要校准**: 必须通过实验找出哪些掩码位对应哪些 SM
3. **粒度问题**: 最小限制粒度可能不是单个 SM，而是 TPC 或 GPC

### 建议

1. **使用 CUDA 13.1+ 的官方 API**:
   ```c
   CUexecAffinityParam affinity;
   affinity.type = CU_EXEC_AFFINITY_TYPE_SM_COUNT;
   affinity.param.smCount.val = target_sms;
   cuCtxCreate_v3(&ctx, &affinity, 1, 0, device);
   ```

2. **如果必须使用 libsmctrl**:
   - 先运行校准脚本找出掩码映射
   - 接受非线性的性能缩放
   - 在较大的 SM 范围内测试（如 8, 16, 24, 32 而不是 64, 65, 66）

3. **验证方法**:
   - 使用 SMID 读取 kernel 确认实际使用的 SM
   - 结合性能测试和 SM ID 探测

## 相关文件

- `calibrate_sm_mask.cu` - 掩码映射校准工具
- `verify_gemm_bench.py` - GEMM 性能验证
- `verify_sm.py` - Element-wise 性能验证
- `vecadd_sm_range.cu` - VecAdd 示例（支持 SM 范围指定）
