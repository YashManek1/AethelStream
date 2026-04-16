//! build.rs — ramflow CUDA detection and kernel compilation
//!
//! Sprint 2 update: compiles overflow_check.cu alongside stub.cu.
//! Both .o files are archived into libramflow_cuda_stub.a.
//!
//! Rules:
//!   • If `--features cuda` (the default): find nvcc, compile kernels → linkable objects.
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
    println!("cargo:rerun-if-changed=kernels/overflow_check.cu");
    println!("cargo:rerun-if-changed=kernels/overflow_check.cuh");
    println!("cargo:rerun-if-changed=kernels/overflow_density.cu");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=NVCC");

    // -----------------------------------------------------------------------
    // 1. Feature mutual-exclusion guard
    // -----------------------------------------------------------------------
    let have_cuda = cfg!(feature = "cuda");
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

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set by Cargo"));

    // -----------------------------------------------------------------------
    // 4. Compile each .cu file to a .o file
    // -----------------------------------------------------------------------
    // Sprint 2 adds overflow_check.cu. All .cu files compile with the same
    // flags. The resulting .o files are archived together into one static lib.
    //
    // -arch=sm_75: Turing minimum (RTX 2000, T4). Ampere (sm_80) and Ada
    //   (sm_89) are backward-compatible with sm_75 PTX. Setting sm_75 means
    //   the PTX JIT will re-optimize for the actual GPU at runtime.
    //
    // --relocatable-device-code=true: Enables separate compilation — the
    //   device code from multiple .cu files can be linked together. Required
    //   when Sprint 4 adds the density kernel that calls helpers from this file.

    let cu_sources: &[(&str, &str)] = &[
        ("kernels/stub.cu", "stub.o"),
        ("kernels/overflow_check.cu", "overflow_check.o"),
        // overflow_density.cu is Sprint 4; uncomment then:
        // ("kernels/overflow_density.cu", "overflow_density.o"),
    ];

    let mut obj_files: Vec<PathBuf> = Vec::new();

    for (src, obj_name) in cu_sources {
        let cu_file = PathBuf::from(src);
        let obj_file = out_dir.join(obj_name);

        // Skip missing files gracefully (e.g., overflow_density.cu is a stub
        // with no real kernel yet — it has a placeholder symbol and compiles,
        // but we don't need to force-compile it until Sprint 4).
        if !cu_file.exists() {
            eprintln!("cargo:warning=Skipping missing CUDA source: {src}");
            continue;
        }

        let status = Command::new(&nvcc)
            .args([
                "--relocatable-device-code=true",
                "--std=c++17",
                "-O3",
                // Minimum target: Turing (sm_75). All newer GPUs are compatible.
                "-arch=sm_75",
                // Include the kernels/ directory so .cu files can #include .cuh headers.
                "-Ikernels",
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
            panic!("ramflow build error: nvcc exited with {status} while compiling {src}");
        }

        obj_files.push(obj_file);
    }

    if obj_files.is_empty() {
        panic!("ramflow build error: no CUDA sources compiled successfully");
    }

    // -----------------------------------------------------------------------
    // 5. Archive all .o files into a single static library
    // -----------------------------------------------------------------------
    // Using one archive keeps the link step simple — Cargo sees one
    // `cargo:rustc-link-lib=static=ramflow_cuda_stub` regardless of how many
    // kernels we add in future sprints.

    let archive = out_dir.join("libramflow_cuda_stub.a");

    // ar crs: c=create, r=insert/replace, s=write symbol table index
    let mut ar_args: Vec<&str> = vec!["crs", archive.to_str().expect("non-UTF-8 path")];
    let obj_strs: Vec<String> = obj_files
        .iter()
        .map(|p| p.to_str().expect("non-UTF-8 path").to_owned())
        .collect();
    for s in &obj_strs {
        ar_args.push(s.as_str());
    }

    let ar_status = Command::new("ar")
        .args(&ar_args)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .unwrap_or_else(|e| panic!("ramflow build error: failed to run `ar`: {e}"));

    if !ar_status.success() {
        panic!("ramflow build error: `ar` failed with status {ar_status}");
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=ramflow_cuda_stub");

    // -----------------------------------------------------------------------
    // 6. Link the CUDA runtime
    // -----------------------------------------------------------------------
    if let Some(cuda_lib) = find_cuda_lib_dir() {
        println!("cargo:rustc-link-search=native={}", cuda_lib.display());
    }
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-cfg=ramflow_real_cuda");
}

// ---------------------------------------------------------------------------
// Helpers (unchanged from Sprint 0)
// ---------------------------------------------------------------------------

/// Return the path to the `nvcc` executable, or `None` if not found.
fn find_nvcc() -> Option<PathBuf> {
    if let Ok(v) = env::var("NVCC") {
        let p = PathBuf::from(v);
        if p.exists() {
            return Some(p);
        }
    }
    for var in &["CUDA_PATH", "CUDA_HOME"] {
        if let Ok(root) = env::var(var) {
            let candidate = PathBuf::from(root).join("bin").join(nvcc_exe());
            if candidate.exists() {
                return Some(candidate);
            }
        }
    }
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

fn nvcc_exe() -> &'static str {
    if cfg!(target_os = "windows") {
        "nvcc.exe"
    } else {
        "nvcc"
    }
}

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
    let fallback = PathBuf::from("/usr/local/cuda/lib64");
    if fallback.exists() {
        return Some(fallback);
    }
    None
}
