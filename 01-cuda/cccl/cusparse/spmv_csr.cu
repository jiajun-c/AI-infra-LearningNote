#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cusparse.h>
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.

using namespace std;

int main() {
    const int A_num_rows = 4;
    const int A_num_cols = 4;
    const int A_nnz = 9;
    int hA_csrOffsets[] = {0, 3, 4, 7, 9};
    int hA_columns[] = {0, 2, 3, 1, 0, 2, 3, 1, 3};
    float hA_values[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};

    float hX[] = {1.0, 2.0, 3.0, 4.0};
    float hY[] = {0.0, 0.0, 0.0, 0.0};
    float     hY_result[]     = { 19.0f, 8.0f, 51.0f, 52.0f };
    float     alpha           = 1.0f;
    float     beta            = 0.0f;
    int   *dA_csrOffsets, *dA_columns;
    float *dA_values, *dX, *dY;

    cudaMalloc((void**)&dA_csrOffsets, (A_num_rows + 1)*sizeof(int));
    cudaMalloc((void**)&dA_columns, A_nnz*sizeof(int));
    cudaMalloc((void**)&dA_values, A_nnz*sizeof(float));
    cudaMalloc((void**)&dX, A_num_cols*sizeof(float));
    cudaMalloc((void**)&dY, A_num_rows*sizeof(float));

    cudaMemcpy(dA_csrOffsets, hA_csrOffsets, (A_num_rows + 1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dA_columns, hA_columns, A_nnz*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dA_values, hA_values, A_nnz*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dX, hX, A_num_cols*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dY, hY, A_num_rows*sizeof(float), cudaMemcpyHostToDevice);

    cusparseHandle_t handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    void* dBuffer = NULL;
    size_t bufSize = 0;
    cusparseCreate(&handle);
    cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,
                      dA_csrOffsets, dA_columns, dA_values,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    cusparseCreateDnVec(&vecX, A_num_cols, dX, CUDA_R_32F);
    cusparseCreateDnVec(&vecY, A_num_rows, dY, CUDA_R_32F);
    cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                          &alpha, matA, vecX, &beta, vecY,
                          CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufSize);
    cudaMalloc(&dBuffer, bufSize);
    cusparseSpMV_preprocess(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
        CUSPARSE_SPMV_ALG_DEFAULT, dBuffer);
    cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer);
    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    cusparseDestroy(handle);
    cudaMemcpy(hY, dY, A_num_rows * sizeof(float),cudaMemcpyDeviceToHost);
    int correct = 1;
    for (int i = 0; i < A_num_rows; i++) {
        if (hY[i] != hY_result[i]) { // direct floating point comparison is not
            correct = 0;             // reliable
            break;
        }
    }
    if (correct)
    printf("spmv_csr_example test PASSED\n");
else
    printf("spmv_csr_example test FAILED: wrong result\n");
}