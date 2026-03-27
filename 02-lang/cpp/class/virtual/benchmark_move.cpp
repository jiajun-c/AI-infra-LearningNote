#include <iostream>
#include <vector>
#include <chrono>
#include <type_traits>

// ============================================================================
// 💣 编译期地雷：强制检查类的移动语义
// ============================================================================
// 只要把这两行宏放在类定义的下方，一旦有队友瞎改析构函数导致移动语义丢失，
// 整个项目将直接编译失败，连运行的机会都不给！
#define ENSURE_MOVE_SEMANTICS(Type) \
    static_assert(std::is_nothrow_move_constructible_v<Type>, \
        "FATAL ERROR: " #Type " lost its noexcept move constructor! Check your destructors.");

// ============================================================================
// 1. 反面教材 (BadOp)：写了虚析构，触发“五法则”陷阱，丢失了 Move
// ============================================================================
class BadOp {
public:
    std::vector<float> weights;

    // 初始化 100 万个 float，约 4MB 内存
    BadOp() : weights(1000000, 1.0f) {} 

    // 致命错误：程序员顺手写了这句，导致编译器拒绝生成移动构造函数
    virtual ~BadOp() = default; 
};

// ❌ 取消下面这行的注释，编译器会立刻报错，因为 BadOp 已经失去了移动能力！
// ENSURE_MOVE_SEMANTICS(BadOp) 

// ============================================================================
// 2. 正面教材 (FastOp)：标准的基础类写法，手动夺回 Move
// ============================================================================
class FastOp {
public:
    std::vector<float> weights;

    FastOp() : weights(1000000, 1.0f) {}

    virtual ~FastOp() = default;

    // 手动补齐五法则，夺回零拷贝能力，并且声明 noexcept
    FastOp(FastOp&&) noexcept = default;
    FastOp& operator=(FastOp&&) noexcept = default;
    
    FastOp(const FastOp&) = default;
    FastOp& operator=(const FastOp&) = default;
};

// ✅ 编译完美通过！这颗地雷没有爆炸。
ENSURE_MOVE_SEMANTICS(FastOp)

// ============================================================================
// ⏱️ 简易高精度计时器框架
// ============================================================================
template <typename Func>
void measure_time(const std::string& test_name, Func&& func) {
    auto start = std::chrono::high_resolution_clock::now();
    
    func(); // 执行测试函数
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    
    std::cout << "[ " << test_name << " ] 耗时: " 
              << duration.count() << " ns" 
              << " (" << duration.count() / 1000000.0 << " ms)\n";
}

// ============================================================================
// 💥 主函数：运行期性能对决
// ============================================================================
int main() {
    std::cout << "正在初始化 4MB 的算子对象...\n";
    BadOp bad_source;
    FastOp fast_source;

    std::cout << "--------------------------------------------------\n";

    // 测试 1：对 BadOp 使用 std::move
    measure_time("BadOp (退化为深拷贝)", [&]() {
        // 表面上写的是 move，底层其实在疯狂 malloc 和 memcpy
        BadOp dest = std::move(bad_source); 
        
        // 防止编译器把 dest 优化掉的黑魔法 (阻断死代码消除)
        asm volatile("" : : "g"(dest.weights.data()) : "memory");
    });

    // 测试 2：对 FastOp 使用 std::move
    measure_time("FastOp (真正的零拷贝)", [&]() {
        // 仅仅交换了底层 vector 的 3 个指针 (总计 24 字节)
        FastOp dest = std::move(fast_source);
        
        asm volatile("" : : "g"(dest.weights.data()) : "memory");
    });

    std::cout << "--------------------------------------------------\n";
    return 0;
}