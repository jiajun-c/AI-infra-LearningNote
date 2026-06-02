#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <sys/epoll.h>
#include <unistd.h>
#include <vector>

struct PipePair {
    int read_fd;
    int write_fd;
};

struct Result {
    int watched_fds;
    int max_fd;
    uint64_t iterations;
    double seconds;
    uint64_t ready_events;
};

void die(const char* message) {
    std::cerr << message << ": " << std::strerror(errno) << "\n";
    std::exit(1);
}

std::vector<PipePair> make_pipes(int count) {
    std::vector<PipePair> pipes;
    pipes.reserve(count);

    for (int i = 0; i < count; ++i) {
        int fds[2];
        if (pipe(fds) != 0) {
            die("pipe");
        }
        pipes.push_back({fds[0], fds[1]});
    }

    return pipes;
}

void close_pipes(const std::vector<PipePair>& pipes) {
    for (const auto& p : pipes) {
        close(p.read_fd);
        close(p.write_fd);
    }
}

Result run_epoll_bench(int watched_fds, uint64_t iterations) {
    auto pipes = make_pipes(watched_fds);

    int epfd = epoll_create1(0);
    if (epfd < 0) {
        die("epoll_create1");
    }

    int max_fd = -1;
    for (size_t i = 0; i < pipes.size(); ++i) {
        epoll_event event{};
        event.events = EPOLLIN;
        event.data.u32 = static_cast<uint32_t>(i);

        if (epoll_ctl(epfd, EPOLL_CTL_ADD, pipes[i].read_fd, &event) != 0) {
            die("epoll_ctl add");
        }

        max_fd = std::max(max_fd, pipes[i].read_fd);
    }

    std::vector<epoll_event> events(16);
    uint64_t ready_events = 0;
    char byte = 'x';
    char buffer = 0;

    auto start = std::chrono::steady_clock::now();

    for (uint64_t i = 0; i < iterations; ++i) {
        const auto& target = pipes.back();

        if (write(target.write_fd, &byte, 1) != 1) {
            die("write");
        }

        int ready = epoll_wait(epfd, events.data(), static_cast<int>(events.size()), -1);
        if (ready < 0) {
            if (errno == EINTR) {
                --i;
                continue;
            }
            die("epoll_wait");
        }

        for (int j = 0; j < ready; ++j) {
            uint32_t index = events[j].data.u32;
            if (read(pipes[index].read_fd, &buffer, 1) != 1) {
                die("read");
            }
            ++ready_events;
        }
    }

    auto end = std::chrono::steady_clock::now();
    double seconds = std::chrono::duration<double>(end - start).count();

    close(epfd);
    close_pipes(pipes);
    return {watched_fds, max_fd, iterations, seconds, ready_events};
}

void print_header() {
    std::cout << std::left << std::setw(12) << "fds"
              << std::right << std::setw(10) << "max_fd"
              << std::setw(14) << "iterations"
              << std::setw(14) << "us/iter"
              << std::setw(14) << "iter/s"
              << std::setw(14) << "ready"
              << "\n";
}

void print_result(const Result& r) {
    double us_per_iter = r.seconds * 1e6 / r.iterations;
    double iter_per_sec = r.iterations / r.seconds;

    std::cout << std::left << std::setw(12) << r.watched_fds
              << std::right << std::setw(10) << r.max_fd
              << std::setw(14) << r.iterations
              << std::setw(14) << std::fixed << std::setprecision(2)
              << us_per_iter
              << std::setw(14) << std::setprecision(0)
              << iter_per_sec
              << std::setw(14) << r.ready_events
              << "\n";
}

int main(int argc, char** argv) {
    uint64_t iterations = 10000;
    std::vector<int> fd_counts = {16, 64, 128, 256, 384, 1024, 4096};

    if (argc >= 2) {
        iterations = std::stoull(argv[1]);
    }

    if (argc >= 3) {
        fd_counts.clear();
        for (int i = 2; i < argc; ++i) {
            fd_counts.push_back(std::stoi(argv[i]));
        }
    }

    std::cout << "epoll benchmark: one ready pipe per iteration\n";
    std::cout << "iterations=" << iterations << "\n\n";

    print_header();
    for (int count : fd_counts) {
        Result r = run_epoll_bench(count, iterations);
        print_result(r);
    }

    return 0;
}
