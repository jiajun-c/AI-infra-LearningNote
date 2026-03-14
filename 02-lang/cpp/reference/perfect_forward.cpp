#include <iostream>

using namespace std;

void func(int& x)  { cout << "Lvalue" << endl; }
void func(int&& x) { cout << "Rvalue" << endl; }

template<typename T>
void wrapper(T&& arg) {
    // arg 本身有一个名字叫 "arg"，所以在 wrapper 内部，arg 永远是左值！
    func(arg);
    func(std::forward<T>(arg)); 
}

int main() {
    int a = 10;
    wrapper(a);  // 传左值 -> T是int& -> arg是int& -> 调用 func(int&) (正确)
    wrapper(10); // 传右值 -> T是int  -> arg是int&& -> 但arg有名字，是左值 -> 调用 func(int&) (错误！！)

    wrapper(std::forward<int>(a));
    wrapper(std::forward<int&&>(10));

}