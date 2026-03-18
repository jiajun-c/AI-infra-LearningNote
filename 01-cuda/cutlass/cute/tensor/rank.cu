#include <cute/tensor.hpp>
#include <stdio.h>

using namespace cute;

int main() {
    // 1. 创建一个 3D 的 shape 变量
    auto shape = make_shape(1, 2, 3);
    
    // 2. CuTe 原生打印，输出格式类似于 (_1,_2,_3) 的 rank
    print(rank(shape)); 
    printf("\n");
    
    // 3. 完美提取原生 int 值！先 decltype 取类型，再喂给 rank_v
    int dim = rank_v<decltype(shape)>;
    printf("%d\n", dim);
    
    // 4. 完美的分支判断！使用编译期 if constexpr
    if constexpr (rank(shape) == Int<3>{}) {
        printf("equal to three\n");
    }

    return 0;
}