#include <mma.h>
#include <cuda_fp16.h>
#include <iostream>
#include <stdio.h>

__global__ void mma_fp16_acc_fp32(half* d_A, half* d_B, float *out) {
  float c[4] = {0., 0., 0., 0.};
  float d[4] = {0., 0., 0., 0.};

  int row = threadIdx.x / 4;
  int col = threadIdx.x % 4;
  int warpID = row * 8 + col;

  half b[4] = {d_B[col*8*2+row], d_B[col*8*2+row + 8], d_B[64+ col*8*2+row], d_B[64+ col*8*2+row+ 8]};
  unsigned const *rA = reinterpret_cast<unsigned const *>(d_A);
  unsigned const *rB = reinterpret_cast<unsigned const *>(b);
  float const *rC = reinterpret_cast<float const *>(&c);
  float *rD = reinterpret_cast<float *>(&d);
  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(rD[0]), "=f"(rD[1]), "=f"(rD[2]), "=f"(rD[3])
      : "r"(rA[warpID]), "r"(rA[warpID+64]), "r"(rA[warpID+4]), "r"(rA[warpID+68]), "r"(rB[0]), "r"(rB[1]),
        "f"(rC[0]), "f"(rC[1]), "f"(rC[2]), "f"(rC[3]));
  memcpy(out + threadIdx.x * 2, rD, 8);
  memcpy(out + 8 * 8 + threadIdx.x * 2, rD + 2, 8);
}

int main() {
  std::cout << "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32" << std::endl;
  half *h_A = (half *)malloc(16 * 16 * sizeof(half));
  half *h_B = (half *)malloc(16 * 8 * sizeof(half));
  float *h_C = (float *)malloc(16 * 8 * sizeof(float));
  for (int i = 0; i < 16; i++) {
    for (int j = 0; j < 16; j++) {
      h_A[i * 16 + j] = j;
    }
  }

  for (int i = 0; i < 16; i++) {
    for (int j = 0; j < 8; j++) {
      h_B[i * 8 + j] = j;
    }
  }
  float *d_C;
  half *d_A;
  half *d_B;
  cudaMalloc(&d_C, 16 * 8 * sizeof(float));
  cudaMalloc(&d_A, 16 * 16 * sizeof(half));
  cudaMalloc(&d_B, 16 * 8 * sizeof(half));
  cudaMemcpy(d_A, h_A, 16 * 16 * sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, 16 * 8 * sizeof(half), cudaMemcpyHostToDevice);
  mma_fp16_acc_fp32<<<1, 32>>>(d_A, d_B, d_C);
  cudaDeviceSynchronize();
  cudaMemcpy(h_C, d_C, 16 * 8 * sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i < 16; i++) {
    for (int j = 0; j < 8; j++) std::cout << h_C[i * 8 + j] << " ";
    std::cout << std::endl;
  }
}