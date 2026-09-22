#include <iostream>
#include <unordered_map>

int main() {
    std::unordered_map<int, int> map;

    std::size_t previous = map.bucket_count();

    for (int i = 0; i < 100; ++i) {
        map.emplace(i, i);

        if (map.bucket_count() != previous) {
            std::cout
                << "size = " << map.size()
                << ", bucket_count: "
                << previous << " -> "
                << map.bucket_count()
                << ", load_factor = "
                << map.load_factor()
                << '\n';

            previous = map.bucket_count();
        }
    }
}