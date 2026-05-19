#include <iostream>
#include <vector>

int main() {
    std::cout << "Case A: push_back triggers reallocation\n";

    std::vector<int> v;
    v.reserve(2);

    v.push_back(10);
    v.push_back(20);

    auto it = v.begin();
    int* p = &v[0];
    auto old_end = v.end();

    std::cout << "before push_back:\n";
    std::cout << "  capacity = " << v.capacity() << '\n';
    std::cout << "  &v[0]    = " << static_cast<void*>(&v[0]) << '\n';
    std::cout << "  p        = " << static_cast<void*>(p) << '\n';

    // capacity 已经满了，这次 push_back 大概率触发扩容
    v.push_back(30);

    std::cout << "after push_back:\n";
    std::cout << "  capacity = " << v.capacity() << '\n';
    std::cout << "  &v[0]    = " << static_cast<void*>(&v[0]) << '\n';
    std::cout << "  p        = " << static_cast<void*>(p)
              << "  <-- old pointer, invalid now\n";

    // 错误示例：不要这样做
    // std::cout << *it << '\n';      // UB
    // std::cout << *p << '\n';       // UB
    // std::cout << *old_end << '\n'; // UB, end 本来也不能解引用

    std::cout << "\nCase B: push_back without reallocation\n";

    std::vector<int> w;
    w.reserve(10);

    w.push_back(1);
    w.push_back(2);

    auto it2 = w.begin();
    int* p2 = &w[0];
    auto old_end2 = w.end();

    std::cout << "before push_back:\n";
    std::cout << "  capacity = " << w.capacity() << '\n';
    std::cout << "  size     = " << w.size() << '\n';
    std::cout << "  &w[0]    = " << static_cast<void*>(&w[0]) << '\n';

    w.push_back(3);

    std::cout << "after push_back:\n";
    std::cout << "  capacity = " << w.capacity() << '\n';
    std::cout << "  size     = " << w.size() << '\n';
    std::cout << "  &w[0]    = " << static_cast<void*>(&w[0]) << '\n';

    std::cout << "  *it2     = " << *it2 << "  <-- still valid\n";
    std::cout << "  *p2      = " << *p2 << "  <-- still valid\n";

    std::cout << "  old_end2 == w.end()? "
              << std::boolalpha << (old_end2 == w.end())
              << "  <-- false, old end iterator is invalid\n";

    std::cout << "\nCase C: erase invalidates iterators at/after erased position\n";

    std::vector<int> x{10, 20, 30, 40, 50};

    auto it_before = x.begin();      // points to 10
    auto it_erased = x.begin() + 2;  // points to 30
    auto it_after = x.begin() + 3;   // points to 40

    std::cout << "before erase: ";
    for (int n : x) std::cout << n << ' ';
    std::cout << '\n';

    auto next = x.erase(it_erased);

    std::cout << "after erase:  ";
    for (int n : x) std::cout << n << ' ';
    std::cout << '\n';

    std::cout << "  *it_before = " << *it_before
              << "  <-- still valid, before erased position\n";
    std::cout << "  *next      = " << *next
              << "  <-- erase returns valid iterator to next element\n";

    // 错误示例：
    // std::cout << *it_erased << '\n'; // UB
    // std::cout << *it_after << '\n';  // UB
}