#include <cuda_runtime.h>

#include <sys/wait.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <iostream>

#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        cudaError_t err__ = (call);                                         \
        if (err__ != cudaSuccess) {                                         \
            std::cerr << "CUDA error: " << cudaGetErrorString(err__)        \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";    \
            std::exit(EXIT_FAILURE);                                        \
        }                                                                   \
    } while (0)

__global__ void init_kernel(float* data, int n, float value) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = value;
    }
}

__global__ void read_peer_kernel(const float* peer, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = peer[i] * 2.0f;
    }
}

__global__ void check_kernel(const float* data, int n, int* errors) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && data[i] != 6.28f) {
        atomicAdd(errors, 1);
    }
}

static void write_all(int fd, const void* buf, size_t bytes) {
    const char* p = static_cast<const char*>(buf);
    while (bytes > 0) {
        ssize_t n = write(fd, p, bytes);
        if (n <= 0) {
            std::perror("write");
            std::exit(EXIT_FAILURE);
        }
        p += n;
        bytes -= static_cast<size_t>(n);
    }
}

static void read_all(int fd, void* buf, size_t bytes) {
    char* p = static_cast<char*>(buf);
    while (bytes > 0) {
        ssize_t n = read(fd, p, bytes);
        if (n == 0) {
            std::cerr << "read: unexpected EOF before receiving IPC handle\n";
            std::exit(EXIT_FAILURE);
        }
        if (n < 0) {
            std::perror("read");
            std::exit(EXIT_FAILURE);
        }
        p += n;
        bytes -= static_cast<size_t>(n);
    }
}

int child_process(int read_fd) {
    cudaIpcMemHandle_t handle;
    read_all(read_fd, &handle, sizeof(handle));
    close(read_fd);

    int device_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    if (device_count < 2) {
        std::cerr << "Need at least 2 GPUs\n";
        return EXIT_FAILURE;
    }

    const int src_device = 0;
    const int dst_device = 1;

    int can_access = 0;
    CHECK_CUDA(cudaDeviceCanAccessPeer(&can_access, dst_device, src_device));
    std::cout << "GPU" << dst_device << " can access GPU" << src_device
              << ": " << can_access << "\n";
    if (!can_access) {
        std::cerr << "P2P is not supported between these GPUs\n";
        return EXIT_FAILURE;
    }

    CHECK_CUDA(cudaSetDevice(dst_device));

    float* peer_ptr = nullptr;
    CHECK_CUDA(cudaIpcOpenMemHandle(
        reinterpret_cast<void**>(&peer_ptr),
        handle,
        cudaIpcMemLazyEnablePeerAccess));

    const int n = 1 << 20;
    const size_t bytes = n * sizeof(float);

    float* d_out = nullptr;
    int* d_errors = nullptr;
    int h_errors = 0;

    CHECK_CUDA(cudaMalloc(&d_out, bytes));
    CHECK_CUDA(cudaMalloc(&d_errors, sizeof(int)));
    CHECK_CUDA(cudaMemset(d_errors, 0, sizeof(int)));

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    read_peer_kernel<<<blocks, threads>>>(peer_ptr, d_out, n);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    check_kernel<<<blocks, threads>>>(d_out, n, d_errors);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaMemcpy(&h_errors, d_errors, sizeof(int), cudaMemcpyDeviceToHost));

    if (h_errors == 0) {
        std::cout << "child: IPC P2P read success\n";
    } else {
        std::cout << "child: IPC P2P read failed, errors = " << h_errors << "\n";
    }

    CHECK_CUDA(cudaIpcCloseMemHandle(peer_ptr));
    CHECK_CUDA(cudaFree(d_out));
    CHECK_CUDA(cudaFree(d_errors));

    return h_errors == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

int parent_process(int write_fd, pid_t child_pid) {
    int device_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    if (device_count < 2) {
        std::cerr << "Need at least 2 GPUs\n";
        return EXIT_FAILURE;
    }

    const int src_device = 0;
    CHECK_CUDA(cudaSetDevice(src_device));

    const int n = 1 << 20;
    const size_t bytes = n * sizeof(float);

    float* d_src = nullptr;
    CHECK_CUDA(cudaMalloc(&d_src, bytes));

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    init_kernel<<<blocks, threads>>>(d_src, n, 3.14f);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaIpcMemHandle_t handle;
    CHECK_CUDA(cudaIpcGetMemHandle(&handle, d_src));

    write_all(write_fd, &handle, sizeof(handle));
    close(write_fd);

    int status = 0;
    waitpid(child_pid, &status, 0);

    CHECK_CUDA(cudaFree(d_src));

    if (WIFEXITED(status) && WEXITSTATUS(status) == EXIT_SUCCESS) {
        std::cout << "parent: child completed successfully\n";
        return EXIT_SUCCESS;
    }

    std::cerr << "parent: child failed\n";
    return EXIT_FAILURE;
}

int main() {
    int pipe_fd[2];
    if (pipe(pipe_fd) != 0) {
        std::perror("pipe");
        return EXIT_FAILURE;
    }

    pid_t pid = fork();
    if (pid < 0) {
        std::perror("fork");
        return EXIT_FAILURE;
    }

    if (pid == 0) {
        close(pipe_fd[1]);
        return child_process(pipe_fd[0]);
    }

    close(pipe_fd[0]);
    return parent_process(pipe_fd[1], pid);
}
