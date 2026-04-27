/*
 * io_uring 裸系统调用 demo（不依赖 liburing）
 * 编译: gcc -O2 io_uring_demo.c -o io_uring_demo
 * 运行: ./io_uring_demo <filename>
 *
 * 与 libaio 版本的核心差异：
 *   - SQ/CQ 是用户态共享内存环形队列，io_submit 退化为纯内存操作，
 *     只有队列为空时才需要一次 io_uring_enter syscall
 *   - 真正的滑动窗口：每收割 1 个立刻补提 1 个，队列深度恒定
 *   - 支持 SQPOLL 模式（本 demo 用默认中断模式，更通用）
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <linux/io_uring.h>

/* ── 参数 ────────────────────────────────────────────────────────────── */
#define READ_SIZE    (4096)
#define QUEUE_DEPTH  (32)
/* ──────────────────────────────────────────────────────────────────────── */

#define DIE(msg) do { perror(msg); exit(1); } while (0)

/* ── 系统调用包装 ─────────────────────────────────────────────────────── */
static int io_uring_setup(unsigned entries, struct io_uring_params *p) {
    return (int)syscall(__NR_io_uring_setup, entries, p);
}
static int io_uring_enter(int fd, unsigned to_submit, unsigned min_complete,
                          unsigned flags) {
    return (int)syscall(__NR_io_uring_enter, fd, to_submit, min_complete,
                        flags, NULL, 0);
}

/* ── 队列描述符 ───────────────────────────────────────────────────────── */
typedef struct {
    int     ring_fd;

    /* SQ（提交队列）映射 */
    uint32_t  sq_entries;
    uint32_t *sq_head, *sq_tail, *sq_mask;
    uint32_t *sq_array;          /* SQ index 数组（指向 sqes） */
    struct io_uring_sqe *sqes;   /* SQE 环（独立 mmap） */

    /* CQ（完成队列）映射 */
    uint32_t  cq_entries;
    uint32_t *cq_head, *cq_tail, *cq_mask;
    struct io_uring_cqe *cqes;   /* CQE 环 */

    void *sq_ring_ptr;  size_t sq_ring_size;   /* 用于 munmap */
    void *cq_ring_ptr;  size_t cq_ring_size;
    void *sqes_ptr;     size_t sqes_size;
} uring_t;

static void uring_init(uring_t *u, unsigned depth) {
    struct io_uring_params p = {0};
    int fd = io_uring_setup(depth, &p);
    if (fd < 0) DIE("io_uring_setup");
    u->ring_fd = fd;

    /* mmap SQ ring（含 sq_array） */
    size_t sq_size = p.sq_off.array + p.sq_entries * sizeof(uint32_t);
    void *sq = mmap(NULL, sq_size, PROT_READ|PROT_WRITE,
                    MAP_SHARED|MAP_POPULATE, fd, IORING_OFF_SQ_RING);
    if (sq == MAP_FAILED) DIE("mmap sq_ring");
    u->sq_ring_ptr  = sq;
    u->sq_ring_size = sq_size;
    u->sq_entries   = p.sq_entries;
    u->sq_head  = (uint32_t *)((char *)sq + p.sq_off.head);
    u->sq_tail  = (uint32_t *)((char *)sq + p.sq_off.tail);
    u->sq_mask  = (uint32_t *)((char *)sq + p.sq_off.ring_mask);
    u->sq_array = (uint32_t *)((char *)sq + p.sq_off.array);

    /* mmap SQE 数组（独立区域） */
    size_t sqes_size = p.sq_entries * sizeof(struct io_uring_sqe);
    void *sqes = mmap(NULL, sqes_size, PROT_READ|PROT_WRITE,
                      MAP_SHARED|MAP_POPULATE, fd, IORING_OFF_SQES);
    if (sqes == MAP_FAILED) DIE("mmap sqes");
    u->sqes_ptr  = sqes;
    u->sqes_size = sqes_size;
    u->sqes      = (struct io_uring_sqe *)sqes;

    /* mmap CQ ring */
    size_t cq_size = p.cq_off.cqes + p.cq_entries * sizeof(struct io_uring_cqe);
    void *cq = mmap(NULL, cq_size, PROT_READ|PROT_WRITE,
                    MAP_SHARED|MAP_POPULATE, fd, IORING_OFF_CQ_RING);
    if (cq == MAP_FAILED) DIE("mmap cq_ring");
    u->cq_ring_ptr  = cq;
    u->cq_ring_size = cq_size;
    u->cq_entries   = p.cq_entries;
    u->cq_head  = (uint32_t *)((char *)cq + p.cq_off.head);
    u->cq_tail  = (uint32_t *)((char *)cq + p.cq_off.tail);
    u->cq_mask  = (uint32_t *)((char *)cq + p.cq_off.ring_mask);
    u->cqes     = (struct io_uring_cqe *)((char *)cq + p.cq_off.cqes);
}

static void uring_destroy(uring_t *u) {
    munmap(u->sq_ring_ptr, u->sq_ring_size);
    munmap(u->cq_ring_ptr, u->cq_ring_size);
    munmap(u->sqes_ptr,    u->sqes_size);
    close(u->ring_fd);
}

/*
 * 把一个 IORING_OP_READ 请求写入 SQ。
 * 返回写入的 SQE tail index（提交前不触发任何 syscall）。
 * user_data 用于在 CQE 里认领请求。
 */
static void uring_submit_read(uring_t *u, int fd, void *buf, size_t len,
                              off_t offset, uint64_t user_data) {
    uint32_t tail  = *u->sq_tail;
    uint32_t idx   = tail & *u->sq_mask;

    struct io_uring_sqe *sqe = &u->sqes[idx];
    memset(sqe, 0, sizeof(*sqe));
    sqe->opcode     = IORING_OP_READ;
    sqe->fd         = fd;
    sqe->addr       = (uint64_t)(uintptr_t)buf;
    sqe->len        = (uint32_t)len;
    sqe->off        = (uint64_t)offset;
    sqe->user_data  = user_data;

    u->sq_array[idx] = idx;         /* SQ index 指向 SQE 槽 */
    __sync_synchronize();           /* 写 tail 前要保证 SQE 对内核可见 */
    *u->sq_tail = tail + 1;
}

/*
 * 通知内核处理 SQ 中积压的请求，并等待至少 min_complete 个 CQE 就绪。
 * to_submit = 当前 SQ 中新增的条目数。
 */
static void uring_flush(uring_t *u, int to_submit, int min_complete) {
    int r = io_uring_enter(u->ring_fd, (unsigned)to_submit,
                           (unsigned)min_complete, 0);
    if (r < 0) DIE("io_uring_enter");
}

/*
 * 从 CQ 消费一个 CQE（非阻塞）。
 * 返回 1 表示成功取到，0 表示 CQ 为空。
 */
static int uring_peek_cqe(uring_t *u, struct io_uring_cqe *out) {
    uint32_t head = *u->cq_head;
    if (head == *u->cq_tail) return 0;
    __sync_synchronize();
    *out = u->cqes[head & *u->cq_mask];
    __sync_synchronize();
    *u->cq_head = head + 1;         /* 告诉内核这个 CQE 已消费 */
    return 1;
}

int main(int argc, char *argv[]) {
    if (argc < 2) { fprintf(stderr, "用法: %s <file>\n", argv[0]); return 1; }

    int fd = open(argv[1], O_RDONLY | O_DIRECT);
    if (fd < 0) DIE("open");

    off_t file_size = lseek(fd, 0, SEEK_END);
    if (file_size <= 0) DIE("lseek");
    long num_blocks = (file_size + READ_SIZE - 1) / READ_SIZE;
    printf("文件大小: %ld 字节，共 %ld 块\n", (long)file_size, num_blocks);

    uring_t u;
    uring_init(&u, QUEUE_DEPTH);

    /* QUEUE_DEPTH 个对齐 buffer 槽，O_DIRECT 要求 4KB 对齐 */
    void *bufs[QUEUE_DEPTH];
    for (int i = 0; i < QUEUE_DEPTH; i++)
        if (posix_memalign(&bufs[i], 4096, READ_SIZE)) DIE("posix_memalign");

    long submitted = 0;
    long completed = 0;
    int  in_flight = 0;   /* 当前飞行中的请求数 */

    while (completed < num_blocks) {

        /* 填满 SQ：把空闲槽全部提交出去 */
        int newly = 0;
        while (submitted < num_blocks && in_flight < QUEUE_DEPTH) {
            int   slot   = submitted % QUEUE_DEPTH;
            off_t offset = (off_t)submitted * READ_SIZE;
            uring_submit_read(&u, fd, bufs[slot], READ_SIZE, offset,
                              (uint64_t)slot);
            submitted++;
            in_flight++;
            newly++;
        }

        /*
         * io_uring_enter：提交 newly 个 SQE，等待至少 1 个 CQE 完成。
         * 与 libaio 的差异：SQE 写入是纯内存操作，enter 只做"通知+等待"。
         */
        if (newly > 0 || in_flight > 0)
            uring_flush(&u, newly, 1);

        /* 消费所有已就绪的 CQE（真正的滑动窗口：收割 1 个就能立刻在下轮补提 1 个） */
        struct io_uring_cqe cqe;
        while (uring_peek_cqe(&u, &cqe)) {
            if (cqe.res < 0) { errno = -cqe.res; DIE("io_uring read"); }
            long slot = (long)cqe.user_data;
            printf("  block %4ld  bytes=%d  first_byte=0x%02x\n",
                   completed, cqe.res,
                   ((unsigned char *)bufs[slot])[0]);
            completed++;
            in_flight--;
        }
    }

    printf("\n全部 %ld 块读取完毕\n", completed);

    uring_destroy(&u);
    for (int i = 0; i < QUEUE_DEPTH; i++) free(bufs[i]);
    close(fd);
    return 0;
}
