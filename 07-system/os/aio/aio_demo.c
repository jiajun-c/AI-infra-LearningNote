/*
 * Linux AIO 练习骨架
 * 编译: gcc -O2 aio_demo.c -o aio_demo -laio
 * 运行: ./aio_demo <filename>
 *
 * 读取策略：把文件拆成若干个 BLOCK_SIZE 的块，批量提交，批量收割，打印每块的首字节。
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <libaio.h>

#define BLOCK_SIZE   (4096)
#define QUEUE_DEPTH  (32)

#define DIE(msg) do { perror(msg); exit(1); } while (0)

int main(int argc, char *argv[]) {
    if (argc < 2) { fprintf(stderr, "用法: %s <file>\n", argv[0]); return 1; }

    /* step 1: 打开文件，O_DIRECT 绕过 page cache，否则 AIO 退化为同步 */
    int fd = open(argv[1], O_RDONLY | O_DIRECT);
    if (fd < 0) DIE("open");

    off_t file_size = lseek(fd, 0, SEEK_END);
    if (file_size <= 0) DIE("lseek");
    long num_blocks = (file_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    printf("文件大小: %ld 字节，共 %ld 块\n", (long)file_size, num_blocks);

    /* step 2: 初始化 AIO 上下文，ctx 必须清零 */
    io_context_t ctx = 0;
    if (io_setup(QUEUE_DEPTH, &ctx) < 0) DIE("io_setup");

    /* step 3: 每个槽分配 4KB 对齐缓冲区，O_DIRECT 要求地址/大小/偏移均对齐 */
    void        *bufs[QUEUE_DEPTH];
    struct iocb  cbs[QUEUE_DEPTH];
    for (int i = 0; i < QUEUE_DEPTH; i++) {
        if (posix_memalign(&bufs[i], 4096, BLOCK_SIZE) != 0) DIE("posix_memalign");
    }

    /* step 4: 主循环：滑动窗口，批量提交 + 批量收割 */
    long submitted = 0;
    long completed = 0;

    while (completed < num_blocks) {

        /* 4a. 填满队列窗口，收集本轮要提交的 iocb 指针 */
        struct iocb *batch[QUEUE_DEPTH];
        int to_submit = 0;
        while (submitted < num_blocks &&
               (submitted - completed) < QUEUE_DEPTH) {
            int   slot   = submitted % QUEUE_DEPTH;
            off_t offset = (off_t)submitted * BLOCK_SIZE;

            io_prep_pread(&cbs[slot], fd, bufs[slot], BLOCK_SIZE, offset);
            cbs[slot].data = (void *)(long)slot;   /* 用于收割时认领 */
            batch[to_submit++] = &cbs[slot];
            submitted++;
        }
        if (to_submit == 0) break;

        /* 4b. 批量提交；返回值是实际入队数，可能小于 to_submit */
        int ret = io_submit(ctx, to_submit, batch);
        if (ret < 0) { errno = -ret; DIE("io_submit"); }

        /* 4c. 收割：等待本轮提交的全部 ret 个请求完成 */
        struct io_event events[QUEUE_DEPTH];
        int collected = 0;
        while (collected < ret) {
            int n = io_getevents(ctx, 1, ret - collected, events + collected, NULL);
            if (n < 0) { errno = -n; DIE("io_getevents"); }
            collected += n;
        }

        for (int i = 0; i < collected; i++) {
            long bytes = events[i].res;
            if (bytes < 0) { errno = (int)-bytes; DIE("aio read"); }
            long slot = (long)events[i].data;
            printf("  block %4ld  bytes=%ld  first_byte=0x%02x\n",
                   completed, bytes,
                   ((unsigned char *)bufs[slot])[0]);
            completed++;
        }
    }

    printf("\n全部 %ld 块读取完毕\n", completed);

    /* step 5: 清理 */
    io_destroy(ctx);
    for (int i = 0; i < QUEUE_DEPTH; i++) free(bufs[i]);
    close(fd);
    return 0;
}
