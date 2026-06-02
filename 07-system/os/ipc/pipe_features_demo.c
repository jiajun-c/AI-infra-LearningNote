#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

static void die(const char *msg)
{
    perror(msg);
    exit(EXIT_FAILURE);
}

static void close_checked(int fd)
{
    if (close(fd) == -1) {
        die("close");
    }
}

static long long now_ms(void)
{
    struct timespec ts;

    if (clock_gettime(CLOCK_MONOTONIC, &ts) == -1) {
        die("clock_gettime");
    }
    return (long long)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

static void wait_child(pid_t pid)
{
    int status;

    if (waitpid(pid, &status, 0) == -1) {
        die("waitpid");
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        fprintf(stderr, "child %ld failed, status=%d\n", (long)pid, status);
        exit(EXIT_FAILURE);
    }
}

static void set_nonblocking(int fd, int enabled)
{
    int flags = fcntl(fd, F_GETFL);

    if (flags == -1) {
        die("fcntl F_GETFL");
    }
    if (enabled) {
        flags |= O_NONBLOCK;
    } else {
        flags &= ~O_NONBLOCK;
    }
    if (fcntl(fd, F_SETFL, flags) == -1) {
        die("fcntl F_SETFL");
    }
}

static void demo_half_duplex(void)
{
    int p[2];
    char ch = 'x';
    ssize_t n;

    puts("\n[1] 半双工: pipe 返回一个读端和一个写端");
    if (pipe(p) == -1) {
        die("pipe");
    }

    n = write(p[0], &ch, 1);
    printf("write(read_end) -> %zd, errno=%d(%s)\n", n, errno, strerror(errno));

    errno = 0;
    n = read(p[1], &ch, 1);
    printf("read(write_end) -> %zd, errno=%d(%s)\n", n, errno, strerror(errno));

    if (write(p[1], &ch, 1) != 1) {
        die("write write_end");
    }
    if (read(p[0], &ch, 1) != 1) {
        die("read read_end");
    }
    printf("write(write_end) + read(read_end) -> ok, data='%c'\n", ch);

    close_checked(p[0]);
    close_checked(p[1]);
}

static void demo_byte_stream(void)
{
    int p[2];
    char buf[16];
    ssize_t n;

    puts("\n[2] 字节流: 多次 write 不保留消息边界");
    if (pipe(p) == -1) {
        die("pipe");
    }

    if (write(p[1], "ABC", 3) != 3) {
        die("write ABC");
    }
    if (write(p[1], "DEF", 3) != 3) {
        die("write DEF");
    }

    n = read(p[0], buf, 4);
    if (n == -1) {
        die("read 4");
    }
    buf[n] = '\0';
    printf("write(\"ABC\") + write(\"DEF\"), read(4) -> \"%s\"\n", buf);

    n = read(p[0], buf, sizeof(buf) - 1);
    if (n == -1) {
        die("read rest");
    }
    buf[n] = '\0';
    printf("second read -> \"%s\"\n", buf);

    close_checked(p[0]);
    close_checked(p[1]);
}

static void demo_anonymous_pipe_inheritance(void)
{
    int p[2];
    pid_t pid;

    puts("\n[3] 亲缘关系: 匿名管道 fd 通过 fork 继承");
    if (pipe(p) == -1) {
        die("pipe");
    }

    pid = fork();
    if (pid == -1) {
        die("fork");
    }
    if (pid == 0) {
        char buf[32];
        ssize_t n;

        close_checked(p[1]);
        n = read(p[0], buf, sizeof(buf) - 1);
        if (n == -1) {
            die("child read");
        }
        buf[n] = '\0';
        printf("child inherited read fd=%d, got \"%s\"\n", p[0], buf);
        close_checked(p[0]);
        _exit(EXIT_SUCCESS);
    }

    close_checked(p[0]);
    if (write(p[1], "from parent", 11) != 11) {
        die("parent write");
    }
    close_checked(p[1]);
    wait_child(pid);
}

static void demo_lifecycle_eof(void)
{
    int p[2];
    pid_t pid;

    puts("\n[4] 生命周期: 所有写端关闭后，读端读到 EOF");
    if (pipe(p) == -1) {
        die("pipe");
    }

    pid = fork();
    if (pid == -1) {
        die("fork");
    }
    if (pid == 0) {
        close_checked(p[0]);
        if (write(p[1], "bye", 3) != 3) {
            die("child write");
        }
        close_checked(p[1]);
        _exit(EXIT_SUCCESS);
    }

    close_checked(p[1]);
    wait_child(pid);

    for (;;) {
        char buf[8];
        ssize_t n = read(p[0], buf, sizeof(buf));

        if (n == -1) {
            die("parent read");
        }
        if (n == 0) {
            puts("read -> 0: EOF, pipe 中没有写端了");
            break;
        }
        printf("read -> %zd bytes\n", n);
    }
    close_checked(p[0]);
}

static void demo_empty_pipe_read_blocks(void)
{
    int p[2];
    pid_t pid;

    puts("\n[5] 阻塞行为 A: 管道空时 read 会阻塞");
    if (pipe(p) == -1) {
        die("pipe");
    }

    pid = fork();
    if (pid == -1) {
        die("fork");
    }
    if (pid == 0) {
        char ch;
        long long start;
        long long end;

        close_checked(p[1]);
        start = now_ms();
        if (read(p[0], &ch, 1) != 1) {
            die("child read empty pipe");
        }
        end = now_ms();
        printf("child read unblocked after about %lld ms, data='%c'\n", end - start, ch);
        close_checked(p[0]);
        _exit(EXIT_SUCCESS);
    }

    close_checked(p[0]);
    sleep(1);
    if (write(p[1], "R", 1) != 1) {
        die("parent write unblock read");
    }
    close_checked(p[1]);
    wait_child(pid);
}

static void demo_full_pipe_write_blocks(void)
{
    int p[2];
    pid_t pid;
    char chunk[4096];
    size_t total = 0;

    puts("\n[6] 阻塞行为 B: 管道满时 write 会阻塞");
    memset(chunk, 'x', sizeof(chunk));
    if (pipe(p) == -1) {
        die("pipe");
    }

    set_nonblocking(p[1], 1);
    for (;;) {
        ssize_t n = write(p[1], chunk, sizeof(chunk));

        if (n > 0) {
            total += (size_t)n;
            continue;
        }
        if (n == -1 && errno == EAGAIN) {
            printf("filled pipe with %zu bytes, next nonblocking write -> EAGAIN\n", total);
            break;
        }
        die("fill pipe");
    }
    set_nonblocking(p[1], 0);

    pid = fork();
    if (pid == -1) {
        die("fork");
    }
    if (pid == 0) {
        long long start = now_ms();
        long long end;

        if (write(p[1], "W", 1) != 1) {
            die("child blocking write");
        }
        end = now_ms();
        printf("child write unblocked after about %lld ms\n", end - start);
        close_checked(p[0]);
        close_checked(p[1]);
        _exit(EXIT_SUCCESS);
    }

    sleep(1);
    if (read(p[0], chunk, sizeof(chunk)) <= 0) {
        die("parent read to free pipe space");
    }
    wait_child(pid);

    close_checked(p[0]);
    close_checked(p[1]);
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);

    demo_half_duplex();
    demo_byte_stream();
    demo_anonymous_pipe_inheritance();
    demo_lifecycle_eof();
    demo_empty_pipe_read_blocks();
    demo_full_pipe_write_blocks();

    puts("\nDone.");
    return 0;
}
