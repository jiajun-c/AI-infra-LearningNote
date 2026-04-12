#include <iostream>
#include <cute/tensor.hpp>

using namespace cute;

void test_slicing() {
    // 1. 定义原始维度 (B, H, N, D)
    auto B = _2{};
    auto H = _4{};
    auto N = _128{};
    auto D = _64{};

    // 2. 创建一个虚拟的 Layout (不分配实际显存，仅验证逻辑)
    auto layout_4d = make_layout(make_shape(B, H, N, D), GenRowMajor{});
    // 这里的 Tensor 实际上不指向有效地址，但 print_layout 只需要 Layout 信息
    auto Q = make_tensor(make_gmem_ptr((float*)0), layout_4d);

    // 定义分块大小
    auto BlockQO = _32{};
    auto HeadDim = _64{};

    // 3. 模拟 Q 的切片: (bx, by, bz, 0)
    // 假设当前 Block 处理 bx=1, by=2, bz=1
    int bx = 1, by = 2, bz = 1;
    auto gQ_all = local_tile(Q, make_shape(_1{}, _1{}, BlockQO, HeadDim), 
                             make_coord(bx, by, bz, 0));
    
    // (0, 0, _, _) 降维
    auto gQ = gQ_all(0, 0, _, _);

    // 4. 模拟 K 的切片: (bx, by, _, 0) -> 产生 Rest 维度
    auto gK_all = local_tile(Q, make_shape(_1{}, _1{}, BlockQO, HeadDim), 
                             make_coord(bx, by, _, 0));
    
    // (0, 0, _, _, _) 降维
    auto gK = gK_all(0, 0, _, _, _);

    // --- 打印结果 ---
    std::cout << "Original Q Layout (B,H,N,D): " << Q.layout() << "\n\n";

    std::cout << "--- Case Q (Single Tile) ---\n";
    std::cout << "gQ_all Rank (Expected 4): " << rank(gQ_all) << " | Layout: " << gQ_all.layout() << "\n";
    std::cout << "gQ final Rank (Expected 2): " << rank(gQ) << " | Layout: " << gQ.layout() << "\n\n";

    std::cout << "--- Case K (Rest Dimension) ---\n";
    std::cout << "gK_all Rank (Expected 5): " << rank(gK_all) << " | Layout: " << gK_all.layout() << "\n";
    std::cout << "gK final Rank (Expected 3): " << rank(gK) << " | Layout: " << gK.layout() << "\n";
    std::cout << "Number of Blocks in gK (RestKV): " << size<2>(gK) << " (Expected 128/32 = 4)\n";
}

int main() {
    test_slicing();
    return 0;
}