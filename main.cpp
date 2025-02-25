#include <iostream>
#include <cmath>
#include <math.h>
#include <float.h>
#include <errno.h>
#include <fenv.h>
#include <chrono>
#include <random>

using namespace std;

void naiveSoftmax(float* dst, float*src, int datalen) {
    float sum = 0;
    for (int i = 0; i < datalen; i++) {
        sum += src[i];
    }
    for (int i = 0; i < datalen; i++) {
        dst[i] = src[i]/sum;
    }
}


void safeSoftmax(float* dst, float*src, int datalen) {
    float max_value = __FLT_MIN__;
    // get max
    for (int i = 0; i < datalen; i++) {
        if (src[i] > max_value) max_value = src[i];
    }
    float sum = 0;

    // get sum
    for (int i = 0; i < datalen; i++) {
        sum += std::exp(src[i] - max_value);
    }

    // compute output
    for (int i = 0; i < datalen; i++) {
        dst[i] = std::exp(src[i] - max_value)/sum;
    }
}

void fastSoftmax(float* dst, float* src, int data_len) {
    float old_max = -FLT_MAX;
    float sum = 0.0f;
    for (int i = 0; i < data_len; i++) {
      float new_max = std::max(old_max, src[i]);
      sum = sum * std::exp(old_max - new_max) + std::exp(src[i] - new_max);
      old_max = new_max;
    }
    for (int i = 0; i < data_len; i++) {
      dst[i] = std::exp(src[i] - old_max) / sum;
    }
    // printf("max = %f, sum = %f\n", old_max, sum);
    //  printf("fastSoftmax Done!\n");
  }
#define N 100000
#define times 10000
int main() {
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    // 创建一个随机数生成器
    std::default_random_engine generator(std::time(nullptr));
    // 定义一个浮点数范围的分布，例如 [0.0, 1.0)
    std::uniform_real_distribution<float> distribution(0.0f, 10.0f);

    float* src = (float*)malloc(N * sizeof(float));
    float* dst = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) src[i] = distribution(generator);
    auto start = std::chrono::system_clock::now();
    for(int i = 0; i < times; i++) naiveSoftmax(dst, src, N);
    auto end = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "naiveSoftmax took " << duration.count() << " milliseconds to execute." << std::endl;
    start = std::chrono::system_clock::now();
    for(int i = 0; i < times; i++) safeSoftmax(dst, src, N);
    end = std::chrono::system_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "safeSoftmax took " << duration.count() << " milliseconds to execute." << std::endl;
    start = std::chrono::system_clock::now();
    for(int i = 0; i < times; i++) fastSoftmax(dst, src, N);
    end = std::chrono::system_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "onlineSoftmax took " << duration.count() << " milliseconds to execute." << std::endl;
}