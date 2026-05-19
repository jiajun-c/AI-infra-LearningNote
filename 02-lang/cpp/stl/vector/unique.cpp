#include <iostream>
#include <vector>
#include <memory>

struct Order
{
    int id;
    int amount;
};

int main() {
    std::vector<std::unique_ptr<Order>>orders;
    orders.push_back(std::make_unique<Order>(Order{1, 100}));
    orders.push_back(std::make_unique<Order>(Order{2, 200}));
    for (const auto& order_ptr : orders) {
        // order_ptr 是对 unique_ptr 的引用
        // 通过 -> 操作符直接访问底层 Order 对象的成员
        std::cout << "Order ID: " << order_ptr->id << '\n'; 
    }

    for (auto it = orders.begin(); it != orders.end(); it++) {
        std::cout << "Order ID: " <<  (*it)->id << '\n';
    }
}
