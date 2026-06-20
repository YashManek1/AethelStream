#![allow(clippy::panic, clippy::expect_used)]

use std::env;
use std::path::PathBuf;

fn main() {
    let is_mock_cuda = cfg!(feature = "mock-cuda");
    let is_cuda = cfg!(feature = "cuda");

    if is_mock_cuda && is_cuda {
        panic!("Features 'mock-cuda' and 'cuda' are mutually exclusive");
    }

    println!("cargo:rerun-if-changed=build.rs");

    if is_mock_cuda {
        println!("cargo:rustc-cfg=doublepass_mock_cuda");
        return;
    }

    if !is_cuda {
        // No CUDA feature active; crate is pure-Rust compatible.
        // No GPU code to compile.
        return;
    }

    // CUDA feature is active: compile .cu kernels
    let cu_sources = vec![
        "kernels/fused_ce.cu",
        "kernels/lora_recompute.cu",
        "kernels/galore_apply.cu",
    ];

    for cu_src in &cu_sources {
        println!("cargo:rerun-if-changed={}", cu_src);
    }

    // Find nvcc
    let nvcc_exe = find_nvcc().expect("nvcc not found; set NVCC env or CUDA_PATH");
    println!(
        "cargo:rustc-link-search=native={}",
        find_cuda_lib_dir().display()
    );

    // Compile each .cu to .o
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    let mut obj_files = Vec::new();

    for cu_src in &cu_sources {
        let obj_file = out_dir.join(
            PathBuf::from(cu_src)
                .file_stem()
                .expect("cu file stem")
                .to_string_lossy()
                .to_string()
                + ".o",
        );

        let status = std::process::Command::new(&nvcc_exe)
            .arg("-c")
            .arg(cu_src)
            .arg("-o")
            .arg(&obj_file)
            .arg("-arch=sm_75")
            .arg("--std=c++17")
            .arg("-O3")
            .arg("-Ikernels")
            .arg("--relocatable-device-code=true")
            .status()
            .expect("nvcc spawn failed");

        if !status.success() {
            panic!("nvcc compilation failed for {}", cu_src);
        }

        obj_files.push(obj_file);
    }

    // Archive object files into libdoublepass_kernels.a
    let lib_path = out_dir.join("libdoublepass_kernels.a");
    let ar_status = std::process::Command::new("ar")
        .arg("rcs")
        .arg(&lib_path)
        .args(&obj_files)
        .status()
        .expect("ar spawn failed");

    if !ar_status.success() {
        panic!("ar archive creation failed");
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=doublepass_kernels");
    println!("cargo:rustc-link-lib=dylib=cudart");
}

/// Find nvcc executable in NVCC env, CUDA_PATH/bin, or PATH.
fn find_nvcc() -> Option<PathBuf> {
    if let Ok(nvcc_env) = env::var("NVCC") {
        let path = PathBuf::from(nvcc_env);
        if path.exists() {
            return Some(path);
        }
    }

    if let Ok(cuda_path) = env::var("CUDA_PATH") {
        let nvcc = PathBuf::from(&cuda_path).join("bin").join("nvcc");
        if nvcc.exists() {
            return Some(nvcc);
        }
    }

    // Try PATH
    if let Ok(path_env) = env::var("PATH") {
        for path_dir in env::split_paths(&path_env) {
            let nvcc = path_dir.join("nvcc");
            if nvcc.exists() {
                return Some(nvcc);
            }
        }
    }

    None
}

/// Find CUDA lib directory (cudart).
fn find_cuda_lib_dir() -> PathBuf {
    if let Ok(cuda_path) = env::var("CUDA_PATH") {
        let lib_dir = PathBuf::from(&cuda_path).join("lib").join("x64");
        if lib_dir.exists() {
            return lib_dir;
        }
        let lib_dir = PathBuf::from(&cuda_path).join("lib");
        if lib_dir.exists() {
            return lib_dir;
        }
    }
    PathBuf::from("/usr/local/cuda/lib64")
}
