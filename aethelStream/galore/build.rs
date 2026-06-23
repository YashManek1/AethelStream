#![allow(clippy::panic, clippy::expect_used)]

use std::env;
use std::path::PathBuf;
use std::process::{Command, Stdio};

fn main() {
    let is_mock_cuda = cfg!(feature = "mock-cuda");
    let is_cuda = cfg!(feature = "cuda");

    if is_mock_cuda && is_cuda {
        panic!("Features 'mock-cuda' and 'cuda' are mutually exclusive");
    }

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=kernels/galore_project.cu");
    println!("cargo:rerun-if-changed=kernels/galore_project.cuh");
    println!("cargo:rerun-if-changed=kernels/quantize_state.cu");
    println!("cargo:rerun-if-changed=kernels/quantize_state.cuh");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=NVCC");

    if is_mock_cuda {
        println!("cargo:rustc-cfg=galore_mock_cuda");
        return;
    }

    if !is_cuda {
        return;
    }

    let cu_sources = [
        "kernels/galore_project.cu",
        "kernels/quantize_state.cu",
    ];

    let nvcc = find_nvcc().expect("nvcc not found; set NVCC env or CUDA_PATH");
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    let mut obj_files = Vec::new();

    for cu_src in &cu_sources {
        let stem = PathBuf::from(cu_src)
            .file_stem()
            .expect("cu stem")
            .to_string_lossy()
            .to_string();
        let obj_file = out_dir.join(format!("{stem}.o"));

        let status = Command::new(&nvcc)
            .args([
                "-c",
                cu_src,
                "-o",
                obj_file.to_str().expect("utf8"),
                "-arch=sm_75",
                "--std=c++17",
                "-O3",
                "-Ikernels",
                "--relocatable-device-code=true",
            ])
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .status()
            .expect("nvcc spawn failed");

        if !status.success() {
            panic!("nvcc compilation failed for {cu_src}");
        }
        obj_files.push(obj_file);
    }

    let lib_path = out_dir.join("libgalore_kernels.a");
    let ar_status = Command::new("ar")
        .arg("rcs")
        .arg(&lib_path)
        .args(&obj_files)
        .status()
        .expect("ar spawn failed");

    if !ar_status.success() {
        panic!("ar archive creation failed");
    }

    if let Some(cuda_lib) = find_cuda_lib_dir() {
        println!("cargo:rustc-link-search=native={}", cuda_lib.display());
    }
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=galore_kernels");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-cfg=galore_real_cuda");
}

fn find_nvcc() -> Option<PathBuf> {
    if let Ok(nvcc_env) = env::var("NVCC") {
        let path = PathBuf::from(nvcc_env);
        if path.exists() {
            return Some(path);
        }
    }
    for var in &["CUDA_PATH", "CUDA_HOME"] {
        if let Ok(root) = env::var(var) {
            let nvcc = PathBuf::from(root).join("bin").join(nvcc_exe());
            if nvcc.exists() {
                return Some(nvcc);
            }
        }
    }
    if Command::new(nvcc_exe())
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
    {
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
    None
}
