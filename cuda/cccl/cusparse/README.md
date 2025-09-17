# cusparse 库使用

cusparse库开源用来计算稀疏矩阵运算，如SPMV和SPMM

## 1. 基础使用

首先需要创建 `cusparseHandle_t` 用于初始化cusparse。

```cpp
cusparseHandle_t handle = NULL;
cusparseCreate(&handle);
```

创建CSR/其他格式的矩阵

```cpp
cusparseSpMatDescr_t matA;
cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,
                      dA_csrOffsets, dA_columns, dA_values,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
```

创建稠密向量

```cpp
cusparseCreateDnVec(&vecX, A_num_cols, dX, CUDA_R_32F);
```

在cusparse12.4 的版本以上，可以使用`cusparseSpMV_preprocess` 去加速SPMV计算

```cpp
cusparseSpMV_preprocess(
    handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
    &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
    CUSPARSE_SPMV_ALG_DEFAULT, dBuffer);
```

进行SPMV计算

```cpp
cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer);
```

## 2. SPMM

使用spGemm的接口可以进行稀疏矩阵乘法的计算。

```cpp
 cusparseSpGEMM_compute(handle, opA, opB,
                        &alpha, matA, matB, &beta, matC,
                        computeType,
                        CUSPARSE_SPGEMM_DEFAULT,
                        spgemmDesc, &bufferSize2, dBuffer2)
```