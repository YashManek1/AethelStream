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
            .arg(cuda_arch().as_str())
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
    archive_objects(&lib_path, &obj_files);

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=doublepass_kernels");
    println!("cargo:rustc-link-lib=dylib=cudart");
}

fn archive_objects(lib_path: &PathBuf, obj_files: &[PathBuf]) {
    use std::process::{Command, Stdio};
    #[cfg(target_os = "windows")]
    {
        let out_arg = format!("/OUT:{}", lib_path.display());
        let ok = Command::new("lib.exe").arg(&out_arg).args(obj_files)
            .stdout(Stdio::null()).stderr(Stdio::null())
            .status().map(|s| s.success()).unwrap_or(false);
        if !ok {
            let s = Command::new("llvm-ar").arg("rcs").arg(lib_path).args(obj_files)
                .status().expect("llvm-ar spawn failed");
            if !s.success() { panic!("archive creation failed (lib.exe and llvm-ar both failed)"); }
        }
    }
    #[cfg(not(target_os = "windows"))]
    {
        let s = Command::new("ar").arg("rcs").arg(lib_path).args(obj_files)
            .status().expect("ar spawn failed");
        if !s.success() { panic!("ar archive creation failed"); }
    }
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
        let nvcc = PathBuf::from(&cuda_path).join("bin").join(nvcc_exe());
        if nvcc.exists() {
            return Some(nvcc);
        }
    }

    // Try PATH
    if std::process::Command::new(nvcc_exe())
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
    {
        return Some(PathBuf::from(nvcc_exe()));
    }

    None
}

fn cuda_arch() -> String {
    let arch = std::env::var("CUDA_ARCH").unwrap_or_else(|_| "sm_75".to_string());
    format!("-arch={arch}")
}

fn nvcc_exe() -> &'static str {
    if cfg!(target_os = "windows") { "nvcc.exe" } else { "nvcc" }
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


