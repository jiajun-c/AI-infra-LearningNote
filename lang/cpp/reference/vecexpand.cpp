#include <iostream>
#include <vector>

using namespace std;

class A {
public:
    A() {}
    // 拷贝构造（慢）
    A(const A&) { cout << "A: Copy Construct (Slow)\n"; }
    // 移动构造（快） - ✅ 加了 noexcept
    A(A&&) noexcept { cout << "A: Move Construct (Fast)\n"; } 
};

class B {
public:
    B() {}
    // 拷贝构造（慢）
    B(const B&) { cout << "B: Copy Construct (Slow)\n"; }
    // 移动构造（快） - ❌ 没加 noexcept
    B(B&&) { cout << "B: Move Construct (Fast)\n"; } 
};

int main() {
    cout << "=== Testing Class A (With noexcept) ===\n";
    vector<A> vecA;
    // 第一次 push，直接构造
    vecA.push_back(A()); 
    cout << "--- Trigger Reallocation for A ---\n";
    // 再次 push，导致 vector 扩容。
    // 因为 A 的移动构造是 noexcept 的，vector 敢放心地把旧元素“移动”到新内存。
    vecA.push_back(A()); 

    cout << "\n\n=== Testing Class B (Without noexcept) ===\n";
    vector<B> vecB;
    vecB.push_back(B());
    cout << "--- Trigger Reallocation for B ---\n";
    // 再次 push，导致 vector 扩容。
    // 因为 B 的移动构造可能抛异常，vector 为了数据安全，不敢用移动，只能“拷贝”旧元素。
    vecB.push_back(B()); 

    return 0;
}