//! CUDA kernel FFI wrappers.

use crate::error::{GaLoreError, Result};
use ramflow::cuda_bridge::CudaStream;

#[cfg(all(feature = "cuda", not(feature = "mock-cuda")))]
mod ffi {
    use std::os::raw::c_void;

    extern "C" {
        pub fn galore_project_forward(
            g: *const c_void,
            p: *const c_void,
            q: *const c_void,
            r: *mut c_void,
            temp: *mut c_void,
            m: i32,
            n: i32,
            r_dim: i32,
            stream: *mut c_void,
        ) -> i32;

        pub fn galore_project_backward(
            n_mat: *const c_void,
            p: *const c_void,
            q: *const c_void,
            g_tilde: *mut c_void,
            temp: *mut c_void,
            m: i32,
            n: i32,
            r_dim: i32,
            stream: *mut c_void,
        ) -> i32;

        pub fn galore_quantize_absmax_f32_to_i8(
            src: *const f32,
            dst: *mut i8,
            scale_out: *mut f32,
            n_elements: i32,
            stream: *mut c_void,
        ) -> i32;

        pub fn galore_dequantize_i8_to_f32(
            src: *const i8,
            dst: *mut f32,
            scale: f32,
            n_elements: i32,
            stream: *mut c_void,
        ) -> i32;
    }
}

/// Launch forward GaLore projection on device (FP16 buffers).
#[allow(clippy::too_many_arguments)]
pub fn project_forward_device(
    g: *const u16,
    p: *const u16,
    q: *const u16,
    r_out: *mut u16,
    temp: *mut u16,
    m: i32,
    n: i32,
    r_dim: i32,
    stream: &CudaStream,
) -> Result<()> {
    if m <= 0 || n <= 0 || r_dim <= 0 {
        return Err(GaLoreError::Shape("invalid projection dims".into()));
    }

    #[cfg(all(feature = "cuda", not(feature = "mock-cuda")))]
    {
        let status = unsafe {
            ffi::galore_project_forward(
                g as *const std::ffi::c_void,
                p as *const std::ffi::c_void,
                q as *const std::ffi::c_void,
                r_out as *mut std::ffi::c_void,
                temp as *mut std::ffi::c_void,
                m,
                n,
                r_dim,
                stream.as_raw(),
            )
        };
        if status != 0 {
            return Err(GaLoreError::Cuda(format!("galore_project_forward status {status}")));
        }
        Ok(())
    }

    #[cfg(not(all(feature = "cuda", not(feature = "mock-cuda"))))]
    {
        let _ = (g, p, q, r_out, temp, stream);
        Err(GaLoreError::Cuda("CUDA not enabled; use CPU project module".into()))
    }
}

/// Mock-device randomized SVD (mock-cuda builds only).
pub fn randomized_svd_device_mock(
    g: &[f32],
    m: usize,
    n: usize,
    cfg: &crate::randomized_svd::RandomizedSvdConfig,
) -> crate::Result<crate::randomized_svd::SubspaceProjections> {
    crate::randomized_svd::randomized_svd_projections(g, m, n, cfg)
}

/// GPU randomized SVD (real CUDA builds).
#[cfg(all(feature = "cuda", not(feature = "mock-cuda")))]
pub fn randomized_svd_device(
    g: &[f32],
    m: usize,
    n: usize,
    cfg: &crate::randomized_svd::RandomizedSvdConfig,
) -> crate::Result<crate::randomized_svd::SubspaceProjections> {
    // TODO: launch cuSOLVER/cuBLAS randomized SVD kernel when available.
    crate::randomized_svd::randomized_svd_projections(g, m, n, cfg)
}

/// Launch backward GaLore projection on device (FP16 buffers).
#[allow(clippy::too_many_arguments)]
pub fn project_backward_device(
    n_mat: *const u16,
    p: *const u16,
    q: *const u16,
    g_tilde: *mut u16,
    temp: *mut u16,
    m: i32,
    n: i32,
    r_dim: i32,
    stream: &CudaStream,
) -> Result<()> {
    #[cfg(all(feature = "cuda", not(feature = "mock-cuda")))]
    {
        let status = unsafe {
            ffi::galore_project_backward(
                n_mat as *const std::ffi::c_void,
                p as *const std::ffi::c_void,
                q as *const std::ffi::c_void,
                g_tilde as *mut std::ffi::c_void,
                temp as *mut std::ffi::c_void,
                m,
                n,
                r_dim,
                stream.as_raw(),
            )
        };
        if status != 0 {
            return Err(GaLoreError::Cuda(format!("galore_project_backward status {status}")));
        }
        Ok(())
    }

    #[cfg(not(all(feature = "cuda", not(feature = "mock-cuda"))))]
    {
        let _ = (n_mat, p, q, g_tilde, temp, m, n, r_dim, stream);
        Err(GaLoreError::Cuda("CUDA not enabled".into()))
    }
}
