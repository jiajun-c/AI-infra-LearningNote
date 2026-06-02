#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sched.h>
#include <string>
#include <vector>
#include <x86intrin.h>

struct alignas(64) Node {
    uint32_t next;
    char padding[60];
};

static_assert(sizeof(Node) == 64);

uint64_t rdtscp() {
    unsigned aux = 0;
    return __rdtscp(&aux);
}

void pin_to_first_allowed_cpu() {
    cpu_set_t current;
    CPU_ZERO(&current);

    if (sched_getaffinity(0, sizeof(current), &current) != 0) {
        return;
    }

    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (CPU_ISSET(cpu, &current)) {
            cpu_set_t target;
            CPU_ZERO(&target);
            CPU_SET(cpu, &target);
            sched_setaffinity(0, sizeof(target), &target);
            return;
        }
    }
}

std::string format_size(size_t bytes) {
    if (bytes < 1024) {
        return std::to_string(bytes) + " B";
    }
    if (bytes < 1024 * 1024) {
        return std::to_string(bytes / 1024) + " KB";
    }
    return std::to_string(bytes / (1024 * 1024)) + " MB";
}

std::vector<Node> make_random_cycle(size_t bytes) {
    size_t count = std::max<size_t>(bytes / sizeof(Node), 1);
    std::vector<Node> nodes(count);

    std::vector<uint32_t> order(count);
    std::iota(order.begin(), order.end(), 0);

    std::mt19937 rng(12345);
    std::shuffle(order.begin(), order.end(), rng);

    for (size_t i = 0; i < count; ++i) {
        nodes[order[i]].next = order[(i + 1) % count];
    }

    return nodes;
}

uint32_t chase(const std::vector<Node>& nodes, uint64_t iterations) {
    uint32_t index = 0;

    for (uint64_t i = 0; i < iterations; ++i) {
        index = nodes[index].next;
    }

    return index;
}

int main() {
    pin_to_first_allowed_cpu();

    std::vector<size_t> sizes = {
        4 * 1024,
        16 * 1024,
        32 * 1024,
        64 * 1024,
        128 * 1024,
        256 * 1024,
        512 * 1024,
        1 * 1024 * 1024,
        2 * 1024 * 1024,
        4 * 1024 * 1024,
        8 * 1024 * 1024,
        16 * 1024 * 1024,
        32 * 1024 * 1024,
        64 * 1024 * 1024,
        128 * 1024 * 1024,
        256 * 1024 * 1024,
    };

    std::cout << std::left
              << std::setw(12) << "working_set"
              << std::right
              << std::setw(14) << "cycles/load"
              << std::setw(14) << "ns/load"
              << std::setw(14) << "loads"
              << "\n";

    volatile uint32_t sink = 0;

    for (size_t size : sizes) {
        auto nodes = make_random_cycle(size);
        size_t actual_size = nodes.size() * sizeof(Node);

        uint64_t iterations = 16ull * 1024 * 1024;
        if (actual_size >= 1ull * 1024 * 1024) {
            iterations = 8ull * 1024 * 1024;
        }
        if (actual_size >= 16ull * 1024 * 1024) {
            iterations = 2ull * 1024 * 1024;
        }

        sink ^= chase(nodes, nodes.size() * 4);

        _mm_mfence();
        uint64_t start_cycles = rdtscp();
        auto start_time = std::chrono::steady_clock::now();

        sink ^= chase(nodes, iterations);

        auto end_time = std::chrono::steady_clock::now();
        uint64_t end_cycles = rdtscp();
        _mm_mfence();

        double cycles_per_load =
            static_cast<double>(end_cycles - start_cycles) / iterations;

        auto elapsed_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                end_time - start_time)
                .count();
        double ns_per_load = static_cast<double>(elapsed_ns) / iterations;

        std::cout << std::left
                  << std::setw(12) << format_size(actual_size)
                  << std::right
                  << std::setw(14) << std::fixed << std::setprecision(2)
                  << cycles_per_load
                  << std::setw(14) << ns_per_load
                  << std::setw(14) << iterations
                  << "\n"
                  << std::flush;
    }

    if (sink == 0xFFFFFFFF) {
        std::cerr << "ignore: " << sink << "\n";
    }

    return 0;
}
