#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <arpa/inet.h>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <netinet/in.h>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <vector>

struct BenchResult {
    std::string name;
    uint64_t messages;
    uint64_t bytes;
    double seconds;
};

void die(const char* message) {
    std::cerr << message << ": " << std::strerror(errno) << "\n";
    std::exit(1);
}

void set_socket_buffer(int fd, int option, int bytes) {
    if (setsockopt(fd, SOL_SOCKET, option, &bytes, sizeof(bytes)) != 0) {
        die("setsockopt");
    }
}

int make_receiver(uint16_t* port) {
    int fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (fd < 0) {
        die("socket receiver");
    }

    set_socket_buffer(fd, SO_RCVBUF, 16 * 1024 * 1024);

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons(0);

    if (bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        die("bind receiver");
    }

    socklen_t len = sizeof(addr);
    if (getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &len) != 0) {
        die("getsockname");
    }

    *port = ntohs(addr.sin_port);
    return fd;
}

int make_sender(uint16_t port) {
    int fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (fd < 0) {
        die("socket sender");
    }

    set_socket_buffer(fd, SO_SNDBUF, 16 * 1024 * 1024);

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons(port);

    if (connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        die("connect sender");
    }

    return fd;
}

void receiver_loop(int fd, std::atomic<bool>& stop, std::atomic<uint64_t>& received) {
    std::vector<char> buffer(2048);

    while (!stop.load(std::memory_order_relaxed)) {
        ssize_t n = recv(fd, buffer.data(), buffer.size(), 0);
        if (n >= 0) {
            received.fetch_add(1, std::memory_order_relaxed);
            continue;
        }

        if (errno == EINTR) {
            continue;
        }
    }
}

BenchResult bench_send(int fd, uint64_t total_messages, size_t payload_size) {
    std::vector<char> payload(payload_size, 'x');

    auto start = std::chrono::steady_clock::now();

    for (uint64_t i = 0; i < total_messages; ++i) {
        ssize_t n = send(fd, payload.data(), payload.size(), 0);
        if (n < 0) {
            if (errno == EINTR) {
                --i;
                continue;
            }
            die("send");
        }
    }

    auto end = std::chrono::steady_clock::now();
    double seconds = std::chrono::duration<double>(end - start).count();

    return {"send", total_messages, total_messages * payload_size, seconds};
}

BenchResult bench_sendmmsg(
    int fd,
    uint64_t total_messages,
    size_t payload_size,
    size_t batch_size) {
    std::vector<std::vector<char>> payloads(batch_size);
    std::vector<iovec> iovecs(batch_size);
    std::vector<mmsghdr> messages(batch_size);

    for (size_t i = 0; i < batch_size; ++i) {
        payloads[i].assign(payload_size, 'x');
        iovecs[i].iov_base = payloads[i].data();
        iovecs[i].iov_len = payloads[i].size();
        messages[i].msg_hdr.msg_iov = &iovecs[i];
        messages[i].msg_hdr.msg_iovlen = 1;
    }

    uint64_t sent = 0;
    auto start = std::chrono::steady_clock::now();

    while (sent < total_messages) {
        size_t want = std::min<uint64_t>(batch_size, total_messages - sent);
        int n = sendmmsg(fd, messages.data(), static_cast<unsigned>(want), 0);

        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            die("sendmmsg");
        }

        sent += static_cast<uint64_t>(n);
    }

    auto end = std::chrono::steady_clock::now();
    double seconds = std::chrono::duration<double>(end - start).count();

    return {"sendmmsg", sent, sent * payload_size, seconds};
}

void print_result(const BenchResult& result) {
    double mps = result.messages / result.seconds / 1e6;
    double mbps = result.bytes / result.seconds / 1024.0 / 1024.0;

    std::cout << std::left << std::setw(10) << result.name
              << std::right << std::setw(14) << result.messages
              << std::setw(14) << std::fixed << std::setprecision(2) << result.seconds
              << std::setw(14) << std::setprecision(2) << mps
              << std::setw(14) << std::setprecision(2) << mbps
              << "\n";
}

int main(int argc, char** argv) {
    uint64_t total_messages = 1'000'000;
    size_t payload_size = 64;
    size_t batch_size = 32;

    if (argc >= 2) {
        total_messages = std::stoull(argv[1]);
    }
    if (argc >= 3) {
        payload_size = std::stoull(argv[2]);
    }
    if (argc >= 4) {
        batch_size = std::stoull(argv[3]);
    }

    if (payload_size == 0 || batch_size == 0) {
        std::cerr << "payload_size and batch_size must be positive\n";
        return 1;
    }

    uint16_t port = 0;
    int receiver_fd = make_receiver(&port);

    std::atomic<bool> stop{false};
    std::atomic<uint64_t> received{0};
    std::thread receiver(receiver_loop, receiver_fd, std::ref(stop), std::ref(received));

    int send_fd = make_sender(port);
    int sendmmsg_fd = make_sender(port);

    std::cout << "UDP localhost throughput, payload=" << payload_size
              << " bytes, messages=" << total_messages
              << ", batch=" << batch_size << "\n\n";

    std::cout << std::left << std::setw(10) << "mode"
              << std::right << std::setw(14) << "messages"
              << std::setw(14) << "seconds"
              << std::setw(14) << "Mmsg/s"
              << std::setw(14) << "MiB/s"
              << "\n";

    BenchResult single = bench_send(send_fd, total_messages, payload_size);
    print_result(single);

    BenchResult batched = bench_sendmmsg(sendmmsg_fd, total_messages, payload_size, batch_size);
    print_result(batched);

    stop.store(true, std::memory_order_relaxed);
    shutdown(receiver_fd, SHUT_RDWR);
    receiver.join();

    close(send_fd);
    close(sendmmsg_fd);
    close(receiver_fd);

    std::cout << "\nreceiver drained about " << received.load() << " packets\n";
    return 0;
}
