// galore_project.cu — GaLore projection kernels using cuBLAS HGEMM
//
// Matrices are stored row-major in device memory.  cuBLAS expects column-major,
// so we use the identity:
//   row_major(C = A @ B)  <=>  col_major(C^T = B^T @ A^T)
//
// Forward:  R = P^T @ G @ Q
//   Step 1: Temp = P^T @ G     (r×n) = (r×m) @ (m×n)
//   Step 2: R    = Temp @ Q    (r×r) = (r×n) @ (n×r)
//
// Backward: G_tilde = P @ N @ Q^T
//   Step 1: Temp    = P @ N       (m×r) = (m×r) @ (r×r)
//   Step 2: G_tilde = Temp @ Q^T  (m×n) = (m×r) @ (r×n)

#include "galore_project.cuh"
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Thread-local cuBLAS handle (created lazily per host thread).
static cublasHandle_t get_cublas_handle() {
    static cublasHandle_t handle = nullptr;
    if (handle == nullptr) {
        cublasCreate(&handle);
    }
    return handle;
}

// Row-major GEMM: C(m×n) = A(m×k) @ B(k×n), all FP16.
// Uses cuBLAS column-major identity: C^T = B^T @ A^T.
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

    // col_major(C^T = B^T @ A^T) with dims (n, m, k)
    return cublasHgemm(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_T,
        n, m, k,
        &alpha,
        B, k,   // B is k×n row-major → lda=k
        A, k,   // A is m×k row-major → lda=k (after transpose view)
        &beta,
        C, n    // C is m×n row-major → ldc=n
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

    // Temp(r×n) = P^T(r×m) @ G(m×n)
    // A = P (m×r), but we need P^T: treat P as B with transpose
    // P^T @ G: A=P transposed → use hgemm with swapped interpretation
    // Direct: Temp = P^T @ G → row_major, m_temp=r, n_temp=n, k=m
    // A_eff = P (stored m×r), we want P^T (r×m) @ G (m×n)
    // Equivalent row_major GEMM: Temp(r×n) = P^T(r×m) @ G(m×n)
    // Using identity: compute G^T @ P in col-major then transpose mentally
    // Simpler: custom two-step with explicit transpose ops via cuBLAS

    // Step 1: Temp(r×n) = P^T @ G
    // P is m×r row-major.  P^T is r×m.
    // cuBLAS: C^T = G^T @ P  →  C = P^T @ G  when C is r×n
    {
        const __half alpha = __float2half(1.0f);
        const __half beta  = __float2half(0.0f);
        cublasSetStream(handle, stream);
        // col_major(Temp^T = G^T @ P)  →  Temp(r×n) = P^T @ G
        cublasStatus_t st = cublasHgemm(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, r, m,
            &alpha,
            G_h, n,   // G^T view: n×m col-major from G(m×n) row-major
            P_h, r,   // P: r×m col-major from P(m×r) row-major
            &beta,
            T_h, n
        );
        if (st != CUBLAS_STATUS_SUCCESS) return static_cast<int>(st);
    }

    // Step 2: R(r×r) = Temp(r×n) @ Q(n×r)
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

    // Step 1: Temp(m×r) = P(m×r) @ N(r×r)
    cublasStatus_t st1 = hgemm_rowmajor(handle, m, r, r, P_h, N_h, T_h, stream);
    if (st1 != CUBLAS_STATUS_SUCCESS) return static_cast<int>(st1);

    // Step 2: G_tilde(m×n) = Temp(m×r) @ Q^T(r×n)
    // Q^T(r×n) from Q(n×r) row-major
    {
        const __half alpha = __float2half(1.0f);
        const __half beta  = __float2half(0.0f);
        cublasSetStream(handle, stream);
        // col_major(G^T = Q @ Temp^T)  →  G = Temp @ Q^T
        cublasStatus_t st = cublasHgemm(
            handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            n, m, r,
            &alpha,
            Q_h, r,
            T_h, r,
            &beta,
            G_h, n
        );
        return (st == CUBLAS_STATUS_SUCCESS) ? 0 : static_cast<int>(st);
    }
}
