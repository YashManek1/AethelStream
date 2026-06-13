//! checksum_shard — compute xxHash3-64 digests for all tensors in shard_index.json.
//!
//! Usage:
//!   checksum_shard --shard-index <path/to/shard_index.json> [--layer <N>]
//!
//! Output: one line per tensor: `layer=<N> tensor=<name> xxh3=0x<hex>`
//!
//! This binary is built only with the `checksums` feature:
//!   cargo build --features checksums --bin checksum_shard

use std::io::Read;
use std::path::PathBuf;

#[cfg(feature = "checksums")]
fn main() {
    use std::process;

    use ramflow::pool::TensorLocationDict;

    let args: Vec<String> = std::env::args().collect();
    let (shard_index_path, layer_filter) = parse_args(&args).unwrap_or_else(|error| {
        eprintln!("error: {error}");
        eprintln!("usage: checksum_shard --shard-index <path> [--layer <N>]");
        process::exit(1);
    });

    let dict = TensorLocationDict::load(&shard_index_path).unwrap_or_else(|error| {
        eprintln!("failed to load shard index: {error}");
        process::exit(1);
    });

    let layer_count = dict.num_layers();
    if layer_count == 0 {
        eprintln!("shard index contains no layers");
        process::exit(1);
    }

    for layer_idx in 0..layer_count as u32 {
        if let Some(filter) = layer_filter {
            if layer_idx != filter {
                continue;
            }
        }
        for tensor_info in dict.tensors_for_layer(layer_idx) {
            let digest =
                read_and_hash(&tensor_info.path, tensor_info.byte_offset, tensor_info.byte_length)
                    .unwrap_or_else(|error| {
                        eprintln!(
                            "failed to read layer={layer_idx} tensor={}: {error}",
                            tensor_info.name
                        );
                        std::process::exit(1);
                    });
            println!(
                "layer={layer_idx} tensor={} xxh3={digest:#018x}",
                tensor_info.name
            );
        }
    }
}

#[cfg(feature = "checksums")]
fn read_and_hash(
    path: &std::path::Path,
    byte_offset: u64,
    byte_length: usize,
) -> Result<u64, Box<dyn std::error::Error>> {
    let mut file = std::fs::File::open(path)?;
    // Seek to the tensor's byte offset inside the shard file.
    std::io::Seek::seek(&mut file, std::io::SeekFrom::Start(byte_offset))?;
    let mut buffer = vec![0u8; byte_length];
    file.read_exact(&mut buffer)?;
    Ok(xxhash_rust::xxh3::xxh3_64(&buffer))
}

#[cfg(feature = "checksums")]
fn parse_args(args: &[String]) -> Result<(PathBuf, Option<u32>), String> {
    let mut shard_index: Option<PathBuf> = None;
    let mut layer_filter: Option<u32> = None;
    let mut index = 1usize;
    while index < args.len() {
        match args[index].as_str() {
            "--shard-index" => {
                index += 1;
                shard_index = Some(PathBuf::from(
                    args.get(index)
                        .ok_or("--shard-index requires a path argument")?,
                ));
            }
            "--layer" => {
                index += 1;
                let raw = args.get(index).ok_or("--layer requires a number")?;
                layer_filter = Some(
                    raw.parse::<u32>()
                        .map_err(|parse_error| format!("--layer value is not a u32: {parse_error}"))?,
                );
            }
            unknown => {
                return Err(format!("unknown argument: {unknown}"));
            }
        }
        index += 1;
    }
    let shard_index = shard_index.ok_or("--shard-index is required")?;
    Ok((shard_index, layer_filter))
}

#[cfg(not(feature = "checksums"))]
fn main() {
    eprintln!("checksum_shard requires the `checksums` feature: cargo build --features checksums --bin checksum_shard");
    std::process::exit(1);
}
