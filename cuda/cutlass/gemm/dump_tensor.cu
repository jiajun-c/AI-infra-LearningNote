#include <iostream>
#include <stdio.h>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/util/host_tensor.h"

// 包含调试工具
#include "cutlass/util/debug.h" 

// 包含核心迭代器组件
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"
#include "cutlass/transform/threadblock/regular_tile_iterator_tensor_op.h"
#include "cutlass/transform/pitch_linear_thread_map.h"

// 定义矩阵大小 (64行 x 32列)
#define ROWS 64
#define COLS 32


int main() {
    using Layout = cutlass::layout::ColumnMajor;
    using Element = float;

    cutlass::HostTensor<Element, Layout> tensor({ROWS, COLS});
}