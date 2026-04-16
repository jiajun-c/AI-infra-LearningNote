#include <iostream>
#include <iomanip>

int main() {
    int threads_per_block = 256;
    int warps_per_block = threads_per_block / 32;  // 8 warps per block
    
    int test_N[] = {64, 64, 128, 128, 256, 256, 512, 1024};
    int test_M[] = {512, 1024, 512, 1024, 512, 1024, 1024, 1024};
    int num_tests = 8;
    
    int sm_counts[] = {16, 33, 66, 108, 132};
    int num_sm_configs = 5;
    
    std::cout << "============================================================" << std::endl;
    std::cout << "GEMV Kernel 负载分析 (threads_per_block = 256, warps_per_block = 8)" << std::endl;
    std::cout << "============================================================" << std::endl;
    
    for (int t = 0; t < num_tests; t++) {
        int N = test_N[t];
        int M = test_M[t];
        
        std::cout << "\n--------------------------------------------------------" << std::endl;
        std::cout << "矩阵大小：" << N << " x " << M << std::endl;
        std::cout << "--------------------------------------------------------" << std::endl;
        
        std::cout << std::left << std::setw(10) << "SM 数量"
                  << std::setw(12) << "总 Warps"
                  << std::setw(15) << "行/Warp"
                  << std::setw(15) << "活跃 Warps"
                  << std::setw(15) << "利用率" << std::endl;
        
        for (int i = 0; i < num_sm_configs; i++) {
            int sms = sm_counts[i];
            int total_warps = sms * warps_per_block;
            float rows_per_warp = (float)N / total_warps;
            int active_warps = (N < total_warps) ? N : total_warps;
            float utilization = (float)active_warps / total_warps * 100;
            
            std::cout << std::left << std::setw(10) << sms
                      << std::setw(12) << total_warps
                      << std::setw(15) << std::fixed << std::setprecision(2) << rows_per_warp
                      << std::setw(15) << active_warps
                      << std::setw(15) << std::fixed << std::setprecision(1) << utilization << "%" << std::endl;
        }
    }
    
    return 0;
}
