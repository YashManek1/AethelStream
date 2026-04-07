//! build.rs — ramflow CUDA detection and kernel compilation
//!
//! Rules:
//!   • If `--features cuda` (the default): find nvcc, compile stub.cu → linkable object.
//!   • If `--features mock-cuda`: skip CUDA entirely, compile with in-tree no-op stubs.
//!   • If BOTH features are active: fail loudly — they are mutually exclusive.
//!   • If neither is active but `mock-cuda` is also absent: fail loudly.
//!
//! We never silently produce a broken binary.

use std::{
    env,
    path::PathBuf,
    process::{Command, Stdio},
};

fn main() {
    // -----------------------------------------------------------------------
    // 0. Re-run triggers
    // -----------------------------------------------------------------------
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=kernels/stub.cu");
    // Also re-run if the user changes CUDA_PATH / CUDA_HOME.
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=NVCC");

    // -----------------------------------------------------------------------
    // 1. Feature mutual-exclusion guard
    // -----------------------------------------------------------------------
    let have_cuda      = cfg!(feature = "cuda");
    let have_mock_cuda = cfg!(feature = "mock-cuda");

    if have_cuda && have_mock_cuda {
        panic!(
            "\n\
            ╔══════════════════════════════════════════════════════════════╗\n\
            ║  ramflow build error: `cuda` and `mock-cuda` are mutually    ║\n\
            ║  exclusive. Enable exactly one of them.                      ║\n\
            ║                                                              ║\n\
            ║  • Real GPU path:   --features cuda       (default)         ║\n\
            ║  • CI / no GPU:     --no-default-features --features mock-cuda ║\n\
            ╚══════════════════════════════════════════════════════════════╝\n"
        );
    }

    // -----------------------------------------------------------------------
    // 2. mock-cuda path — no NVCC required
    // -----------------------------------------------------------------------
    if have_mock_cuda {
        println!("cargo:rustc-cfg=ramflow_mock_cuda");
        // Nothing to link; the Rust stubs handle everything.
        return;
    }

    // -----------------------------------------------------------------------
    // 3. Real CUDA path
    // -----------------------------------------------------------------------
    // 3a. Locate nvcc
    let nvcc = find_nvcc().unwrap_or_else(|| {
        panic!(
            "\n\
            ╔══════════════════════════════════════════════════════════════╗\n\
            ║  ramflow build error: `cuda` feature is enabled but `nvcc`   ║\n\
            ║  was not found.                                              ║\n\
            ║                                                              ║\n\
            ║  Options:                                                    ║\n\
            ║    (a) Install CUDA Toolkit and ensure nvcc is on PATH.      ║\n\
            ║    (b) Set NVCC=/path/to/nvcc in the environment.            ║\n\
            ║    (c) Set CUDA_PATH or CUDA_HOME to your CUDA install root.  ║\n\
            ║    (d) Build without a GPU:                                  ║\n\
            ║          cargo build --no-default-features --features mock-cuda ║\n\
            ╚══════════════════════════════════════════════════════════════╝\n"
        )
    });

    // 3b. Identify output directory (Cargo sets OUT_DIR)
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set by Cargo"));

    // 3c. Compile stub.cu → stub.o
    let cu_file = PathBuf::from("kernels/stub.cu");
    let obj_file = out_dir.join("stub.o");

    let status = Command::new(&nvcc)
        .args([
            // Produce relocatable device code so downstream kernels can link.
            "--relocatable-device-code=true",
            // Match the host compiler's C++ standard
            "--std=c++17",
            // Compile only — do not link
            "-c",
            cu_file.to_str().expect("non-UTF-8 path"),
            "-o",
            obj_file.to_str().expect("non-UTF-8 path"),
        ])
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .unwrap_or_else(|e| {
            panic!("ramflow build error: failed to spawn nvcc ({nvcc:?}): {e}")
        });

    if !status.success() {
        panic!(
            "ramflow build error: nvcc exited with status {status} while compiling kernels/stub.cu"
        );
    }

    // 3d. Tell Cargo to link stub.o
    //   We use a static-lib approach: ar the .o into a thin archive and
    //   point cargo:rustc-link-search at OUT_DIR.
    let archive = out_dir.join("libramflow_cuda_stub.a");
    let ar_status = Command::new("ar")
        .args([
            "crs",
            archive.to_str().expect("non-UTF-8 path"),
            obj_file.to_str().expect("non-UTF-8 path"),
        ])
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .unwrap_or_else(|e| panic!("ramflow build error: failed to run `ar`: {e}"));

    if !ar_status.success() {
        panic!("ramflow build error: `ar` failed with status {ar_status}");
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=ramflow_cuda_stub");

    // 3e. Also link the CUDA runtime itself
    if let Some(cuda_lib) = find_cuda_lib_dir() {
        println!("cargo:rustc-link-search=native={}", cuda_lib.display());
    }
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-cfg=ramflow_real_cuda");
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Return the path to the `nvcc` executable, or `None` if not found.
///
/// Search order:
///   1. `NVCC` environment variable
///   2. `{CUDA_PATH}/bin/nvcc`  (Windows convention)
///   3. `{CUDA_HOME}/bin/nvcc`  (Linux/macOS convention)
///   4. Plain `nvcc` on PATH
fn find_nvcc() -> Option<PathBuf> {
    // 1. Explicit override
    if let Ok(v) = env::var("NVCC") {
        let p = PathBuf::from(v);
        if p.exists() {
            return Some(p);
        }
    }

    // 2/3. CUDA_PATH / CUDA_HOME
    for var in &["CUDA_PATH", "CUDA_HOME"] {
        if let Ok(root) = env::var(var) {
            let candidate = PathBuf::from(root).join("bin").join(nvcc_exe());
            if candidate.exists() {
                return Some(candidate);
            }
        }
    }

    // 4. PATH probe via `which nvcc`
    let probe = Command::new(nvcc_exe())
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
    if probe.map(|s| s.success()).unwrap_or(false) {
        return Some(PathBuf::from(nvcc_exe()));
    }

    None
}

/// Platform-aware executable name.
fn nvcc_exe() -> &'static str {
    if cfg!(target_os = "windows") {
        "nvcc.exe"
    } else {
        "nvcc"
    }
}

/// Try to find the CUDA library directory (where libcudart lives).
fn find_cuda_lib_dir() -> Option<PathBuf> {
    for var in &["CUDA_PATH", "CUDA_HOME"] {
        if let Ok(root) = env::var(var) {
            let lib_dir = if cfg!(target_os = "windows") {
                PathBuf::from(&root).join("lib").join("x64")
            } else {
                PathBuf::from(&root).join("lib64")
            };
            if lib_dir.exists() {
                return Some(lib_dir);
            }
        }
    }
    // Common hard-coded Linux fallback
    let fallback = PathBuf::from("/usr/local/cuda/lib64");
    if fallback.exists() {
        return Some(fallback);
    }
    None
}
