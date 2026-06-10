fn main() {
    // Link cuFile when `gds` feature is enabled
    #[cfg(feature = "gds")]
    {
        let cuda_home = std::env::var("CUDA_HOME")
            .or_else(|_| std::env::var("CUDA_PATH"))
            .unwrap_or_else(|_| "/usr/local/cuda".to_string());

        println!("cargo:rustc-link-search=native={}/lib64", cuda_home);
        println!("cargo:rustc-link-lib=dylib=cufile");
    }
}
