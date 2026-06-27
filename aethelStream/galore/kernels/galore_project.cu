// galore_project.cu -- GaLore projection kernels using cuBLAS HGEMM
//
// Matrices are stored row-major in device memory.  cuBLAS expects column-major.
// Standard identity for row-major C(m*n) = A(m*k) @ B(k*n):
//
//   C_col(n*m) = B_col(n*k) @ A_col(k*m)   [OP_N OP_N]
//
// where X_col is X_rm reinterpreted as col-major (gives X_rm^T).
//
// Forward:  R = P^T @ G @ Q
//   Step 1: Temp(r*n) = P^T @ G  -- custom: OP_N G(m*n), OP_T P(m*r)
//   Step 2: R(r*r)    = Temp @ Q -- hgemm_rowmajor(r,r,n, Temp,Q)
//
// Backward: G_tilde = P @ N @ Q^T
//   Step 1: Temp(m*r)    = P @ N       -- hgemm_rowmajor(m,r,r, P,N)
//   Step 2: G_tilde(m*n) = Temp @ Q^T  -- custom: OP_T Q(n*r), OP_N Temp(m*r)

#include "galore_project.cuh"
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Per-thread cuBLAS handle (thread_local avoids races on multi-threaded hosts).
static cublasHandle_t get_cublas_handle() {
    static thread_local cublasHandle_t handle = nullptr;
    if (handle == nullptr) {
        cublasCreate(&handle);
    }
    return handle;
}

// Row-major GEMM: C(m*n) = A(m*k) @ B(k*n), all FP16.
//
// C_rm(m*n) = A_rm(m*k) @ B_rm(k*n)
// <=>  C_col(n*m) = B_col(n*k) @ A_col(k*m)   [OP_N OP_N]
//
// B_rm(k*n) as col-major n*k with ldb=n  (ldb >= n check)
// A_rm(m*k) as col-major k*m with lda=k  (lda >= k check)
// C_rm(m*n) as col-major n*m with ldc=n  (ldc >= n check)
static cublasStatus_t hgemm_rowmajor(
    cublasHandle_t handle,
    int m, int n, int k,
    const __half* A,
    const __half* B,
    __half* C,
    cudaStream_t stream
) {
    cublasSetStream(handle, stream);
    const __half alpha = __float2half(1.0f);
    const __half beta  = __float2half(0.0f);
    return cublasHgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        n, m, k,
        &alpha,
        B, n,   // B_rm(k*n) as col-major n*k,  ldb=n
        A, k,   // A_rm(m*k) as col-major k*m,  lda=k
        &beta,
        C, n    // C_rm(m*n) as col-major n*m,  ldc=n
    );
}

extern "C" int galore_project_forward(
    const void* G,
    const void* P,
    const void* Q,
    void*       R,
    void*       temp,
    int         m,
    int         n,
    int         r,
    cudaStream_t stream
) {
    if (m <= 0 || n <= 0 || r <= 0) return -1;

    const __half* G_h = static_cast<const __half*>(G);
    const __half* P_h = static_cast<const __half*>(P);
    const __half* Q_h = static_cast<const __half*>(Q);
    __half*       R_h = static_cast<__half*>(R);
    __half*       T_h = static_cast<__half*>(temp);

    cublasHandle_t handle = get_cublas_handle();

    // Step 1: Temp(r*n) = P^T(r*m) @ G(m*n)
    //
    // Compute Temp^T(n*r) = G^T(n*m) @ P(m*r) in col-major.
    // G_rm(m*n) as col-major n*m with lda=n => G_rm^T; OP_N (lda >= n check)
    // P_rm(m*r) as col-major r*m with lda=r => P_rm^T; OP_T => P_rm (m*r) (lda >= r check)
    // Result Temp^T(n*r) col-major with ldc=n
    {
        const __half alpha = __float2half(1.0f);
        const __half beta  = __float2half(0.0f);
        cublasSetStream(handle, stream);
        cublasStatus_t st = cublasHgemm(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            n, r, m,
            &alpha,
            G_h, n,   // G_rm(m*n) col-major n*m, lda=n
            P_h, r,   // P_rm(m*r) col-major r*m, lda=r; OP_T => P_rm
            &beta,
            T_h, n    // Temp^T(n*r) col-major, ldc=n
        );
        if (st != CUBLAS_STATUS_SUCCESS) return static_cast<int>(st);
    }

    // Step 2: R(r*r) = Temp(r*n) @ Q(n*r)
    cublasStatus_t st2 = hgemm_rowmajor(handle, r, r, n, T_h, Q_h, R_h, stream);
    return (st2 == CUBLAS_STATUS_SUCCESS) ? 0 : static_cast<int>(st2);
}

extern "C" int galore_project_backward(
    const void* N,
    const void* P,
    const void* Q,
    void*       G_tilde,
    void*       temp,
    int         m,
    int         n,
    int         r,
    cudaStream_t stream
) {
    if (m <= 0 || n <= 0 || r <= 0) return -1;

    const __half* N_h  = static_cast<const __half*>(N);
    const __half* P_h  = static_cast<const __half*>(P);
    const __half* Q_h  = static_cast<const __half*>(Q);
    __half*       G_h  = static_cast<__half*>(G_tilde);
    __half*       T_h  = static_cast<__half*>(temp);

    cublasHandle_t handle = get_cublas_handle();

    // Step 1: Temp(m*r) = P(m*r) @ N(r*r)
    cublasStatus_t st1 = hgemm_rowmajor(handle, m, r, r, P_h, N_h, T_h, stream);
    if (st1 != CUBLAS_STATUS_SUCCESS) return static_cast<int>(st1);

    // Step 2: G_tilde(m*n) = Temp(m*r) @ Q^T(r*n)
    //
    // G^T(n*m) = Q(n*r) @ Temp^T(r*m) in col-major.
    // Q_rm(n*r) col-major r*n, lda=r; OP_T => Q_rm (n*r) (lda >= r check)
    // T_rm(m*r) col-major r*m, lda=r; OP_N => T_rm^T (r*m) (lda >= r check)
    // Result G^T(n*m) col-major, ldc=n
    {
        const __half alpha = __float2half(1.0f);
        const __half beta  = __float2half(0.0f);
        cublasSetStream(handle, stream);
        cublasStatus_t st = cublasHgemm(
            handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            n, m, r,
            &alpha,
            Q_h, r,   // Q_rm(n*r) col-major r*n, lda=r; OP_T => Q_rm
            T_h, r,   // T_rm(m*r) col-major r*m, lda=r; OP_N => T_rm^T
            &beta,
            G_h, n    // G^T(n*m) col-major, ldc=n
        );
        return (st == CUBLAS_STATUS_SUCCESS) ? 0 : static_cast<int>(st);
    }
}
