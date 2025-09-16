#include <mma.h>
#include <cuda_fp16.h>
#include <iostream>
#include <stdio.h>

__global__ void mma_fp16_acc_fp32(float *out) {
  float c[4] = {0., 0., 0., 0.};
  float d[4] = {0., 0., 0., 0.};
  half a[8] = {1., 2., 1., 2., 1., 2., 1., 2.};
  half b[4] = {1., 2., 3., 4.};
  unsigned const *rA = reinterpret_cast<unsigned const *>(&a);
  unsigned const *rB = reinterpret_cast<unsigned const *>(&b);
  float const *rC = reinterpret_cast<float const *>(&c);
  float *rD = reinterpret_cast<float *>(&d);
  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(rD[0]), "=f"(rD[1]), "=f"(rD[2]), "=f"(rD[3])
      : "r"(rA[0]), "r"(rA[1]), "r"(rA[2]), "r"(rA[3]), "r"(rB[0]), "r"(rB[1]),
        "f"(rC[0]), "f"(rC[1]), "f"(rC[2]), "f"(rC[3]));
  printf("%f\n", rD[0]);
  memcpy(out + threadIdx.x * 2, rD, 8);
  memcpy(out + 8 * 8 + threadIdx.x * 2, rD + 2, 8);
}

int main() {
  std::cout << "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32" << std::endl;
  float *h_C = (float *)malloc(16 * 8 * sizeof(float));
  float *d_C;
  cudaMalloc(&d_C, 16 * 8 * sizeof(float));
  mma_fp16_acc_fp32<<<1, 32>>>(d_C);
  cudaDeviceSynchronize();
  cudaMemcpy(h_C, d_C, 16 * 8 * sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i < 16; i++) {
    for (int j = 0; j < 8; j++) std::cout << h_C[i * 8 + j] << " ";
    std::cout << std::endl;
  }
}