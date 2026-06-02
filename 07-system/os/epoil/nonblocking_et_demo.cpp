#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <string>
#include <sys/epoll.h>
#include <unistd.h>

void die(const char* message) {
    std::cerr << message << ": " << std::strerror(errno) << "\n";
    std::exit(1);
}

void set_nonblocking(int fd) {
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0) {
        die("fcntl F_GETFL");
    }

    if (fcntl(fd, F_SETFL, flags | O_NONBLOCK) != 0) {
        die("fcntl F_SETFL O_NONBLOCK");
    }
}

void write_all(int fd, const std::string& data) {
    const char* p = data.data();
    size_t left = data.size();

    while (left > 0) {
        ssize_t n = write(fd, p, left);
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            die("write");
        }
        p += n;
        left -= static_cast<size_t>(n);
    }
}

void demo_nonblocking_read_empty_pipe() {
    int fds[2];
    if (pipe(fds) != 0) {
        die("pipe");
    }

    set_nonblocking(fds[0]);

    char byte = 0;
    ssize_t n = read(fds[0], &byte, 1);

    std::cout << "[1] read empty nonblocking pipe\n";
    if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
        std::cout << "read returned immediately: -1, errno=EAGAIN\n\n";
    } else {
        std::cout << "unexpected read result: " << n << "\n\n";
    }

    close(fds[0]);
    close(fds[1]);
}

int make_epoll_for_read_fd(int read_fd) {
    int epfd = epoll_create1(0);
    if (epfd < 0) {
        die("epoll_create1");
    }

    epoll_event event{};
    event.events = EPOLLIN | EPOLLET;
    event.data.fd = read_fd;

    if (epoll_ctl(epfd, EPOLL_CTL_ADD, read_fd, &event) != 0) {
        die("epoll_ctl add");
    }

    return epfd;
}

void demo_et_read_only_once() {
    int fds[2];
    if (pipe(fds) != 0) {
        die("pipe");
    }

    set_nonblocking(fds[0]);
    int epfd = make_epoll_for_read_fd(fds[0]);

    write_all(fds[1], "abcdefghij");

    epoll_event event{};
    int n = epoll_wait(epfd, &event, 1, 1000);
    if (n <= 0) {
        die("epoll_wait first event");
    }

    char byte = 0;
    ssize_t r = read(fds[0], &byte, 1);
    if (r != 1) {
        die("read one byte");
    }

    std::cout << "[2] EPOLLET but read only one byte\n";
    std::cout << "first epoll_wait notified, read byte='" << byte << "'\n";

    n = epoll_wait(epfd, &event, 1, 200);
    std::cout << "second epoll_wait timeout result=" << n
              << " while 9 bytes are still buffered\n\n";

    close(epfd);
    close(fds[0]);
    close(fds[1]);
}

void demo_et_drain_until_eagain() {
    int fds[2];
    if (pipe(fds) != 0) {
        die("pipe");
    }

    set_nonblocking(fds[0]);
    int epfd = make_epoll_for_read_fd(fds[0]);

    write_all(fds[1], "abcdefghij");

    epoll_event event{};
    int n = epoll_wait(epfd, &event, 1, 1000);
    if (n <= 0) {
        die("epoll_wait first event");
    }

    std::string drained;
    char buffer[4];

    while (true) {
        ssize_t r = read(fds[0], buffer, sizeof(buffer));
        if (r > 0) {
            drained.append(buffer, buffer + r);
            continue;
        }

        if (r < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            break;
        }

        if (r < 0 && errno == EINTR) {
            continue;
        }

        die("read drain");
    }

    std::cout << "[3] EPOLLET and drain until EAGAIN\n";
    std::cout << "drained data='" << drained << "'\n";
    std::cout << "read stopped at EAGAIN, so the pipe is empty now\n";

    n = epoll_wait(epfd, &event, 1, 200);
    std::cout << "next epoll_wait timeout result=" << n << "\n\n";

    close(epfd);
    close(fds[0]);
    close(fds[1]);
}

int main() {
    demo_nonblocking_read_empty_pipe();
    demo_et_read_only_once();
    demo_et_drain_until_eagain();
    return 0;
}
