// src/allocator/numa.rs -- NUMA topology detection and page-binding
//
// Feature gate: `numa`.  Platform gate: `target_os = "linux"`.
//
// On NUMA systems, physical memory is organised in nodes each with its own
// memory controller.  A CPU or GPU PCIe root attached to node N accesses node
// N's RAM without crossing the inter-node interconnect (HyperTransport /
// AMD Infinity Fabric / Intel UPI).  Binding pool buffers to the GPU's NUMA
// node reduces cross-node PCIe traffic and can recover 5-15% effective
// bandwidth on dual-socket or dual-CCD machines.
//
// On single-socket consumer hardware (the common case), the GPU's
// /sys/bus/pci/devices/<addr>/numa_node reads -1.  In that case this module
// returns NumaConfig::disabled() and no mbind calls are made — it is a
// complete no-op with zero runtime overhead.
//
// Drop constraint: mbind does not change how memory was allocated; Drop still
// calls free_aligned or munmap_huge exactly as before.  NUMA binding is a
// kernel hint about physical page placement, not a second allocator.

// ===========================================================================
// NumaConfig — public data type, always compiled
// ===========================================================================

/// NUMA topology result from [`detect`].
///
/// Always defined so non-NUMA code paths can store and pass it without
/// needing `#[cfg]` gating at every call site.
#[derive(Debug, Clone, Copy)]
pub struct NumaConfig {
    /// NUMA node hosting the GPU's PCIe root complex.
    /// `None` on single-socket systems or when detection fails.
    pub gpu_node: Option<u32>,
    /// `true` when NUMA is available and `mbind_buffer` calls are safe.
    /// Always `false` without the `numa` feature or on non-Linux.
    pub available: bool,
}

impl NumaConfig {
    /// No-op config used when NUMA is unavailable.
    pub fn disabled() -> Self {
        NumaConfig {
            gpu_node: None,
            available: false,
        }
    }
}

impl Default for NumaConfig {
    fn default() -> Self {
        Self::disabled()
    }
}

// ===========================================================================
// detect() -- topology probe
// ===========================================================================

/// Detect the GPU NUMA node by reading
/// `/sys/bus/pci/devices/<pci_addr>/numa_node`.
///
/// `pci_addr` is a string like `"0000:01:00.0"`.  Pass `None` to scan
/// `/sys/bus/pci/devices/` for the first display/3D-controller class device
/// (PCI class `0x0302xx` or `0x0300xx`).
///
/// Returns [`NumaConfig::disabled`] on any I/O error, when the node reads
/// `-1` (single-socket), or on non-Linux / without the `numa` feature.
#[cfg(all(feature = "numa", target_os = "linux"))]
pub fn detect(pci_addr: Option<&str>) -> NumaConfig {
    let node = match pci_addr {
        Some(addr) => read_numa_node_for(addr),
        None => scan_gpu_numa_node(),
    };
    match node {
        Some(n) => NumaConfig {
            gpu_node: Some(n),
            available: true,
        },
        None => NumaConfig::disabled(),
    }
}

/// Stub so call sites compile on non-Linux or without the `numa` feature.
#[cfg(not(all(feature = "numa", target_os = "linux")))]
pub fn detect(_pci_addr: Option<&str>) -> NumaConfig {
    NumaConfig::disabled()
}

// ===========================================================================
// mbind_buffer() -- bind a buffer's pages to a NUMA node
// ===========================================================================

/// Bind the memory region `[ptr, ptr+size)` to `node` using
/// `mbind(MPOL_BIND, MPOL_MF_MOVE)`.
///
/// The request is rounded to OS page boundaries (4096 bytes) as required by
/// the kernel.  The buffer's 512-byte O_DIRECT alignment is unaffected —
/// mbind is a policy hint, not a re-allocation.
///
/// Returns `true` on success.  Returns `false` without panicking if:
/// - `mbind` returns `EPERM` (non-root, seccomp filter).
/// - `node` exceeds the system's node count.
/// - Any other kernel error.
///
/// A `false` return is a performance degradation, not a correctness error.
/// Pool startup logs a note but continues training normally.
#[cfg(all(feature = "numa", target_os = "linux"))]
pub fn mbind_buffer(ptr: *mut u8, size: usize, node: u32) -> bool {
    if size == 0 {
        return true;
    }

    // Round address down and end up to page granularity.
    const PAGE: usize = 4096;
    let start = (ptr as usize) & !(PAGE - 1);
    let end = ((ptr as usize).saturating_add(size).saturating_add(PAGE - 1)) & !(PAGE - 1);
    let len = end - start;

    // nodemask: bit n = 1 means "allow node n".
    // maxnode = 64: covers nodes 0-63 (more than any real system has).
    let nodemask: libc::c_ulong = 1 << (node as libc::c_ulong);
    let maxnode: libc::c_ulong = 64;

    // MPOL_BIND = 2: pages are strictly bound to nodemask nodes.
    // MPOL_MF_MOVE = 2: migrate already-faulted pages to the target node.
    const MPOL_BIND: libc::c_int = 2;
    const MPOL_MF_MOVE: libc::c_uint = 2;

    // SAFETY: `start` is page-aligned, `len` is a multiple of PAGE_SIZE,
    // `nodemask` is a valid stack-local c_ulong.  mbind does not take
    // ownership of the pointer and does not alias the buffer contents.
    let rc = unsafe {
        libc::mbind(
            start as *mut libc::c_void,
            len as libc::size_t,
            MPOL_BIND,
            &nodemask as *const libc::c_ulong,
            maxnode,
            MPOL_MF_MOVE,
        )
    };
    rc == 0
}

/// Stub so non-Linux callers compile.  Always returns `false`.
#[cfg(not(all(feature = "numa", target_os = "linux")))]
pub fn mbind_buffer(_ptr: *mut u8, _size: usize, _node: u32) -> bool {
    false
}

// ===========================================================================
// Private helpers (Linux + numa only)
// ===========================================================================

#[cfg(all(feature = "numa", target_os = "linux"))]
fn read_numa_node_for(pci_addr: &str) -> Option<u32> {
    let path = format!("/sys/bus/pci/devices/{pci_addr}/numa_node");
    let text = std::fs::read_to_string(path).ok()?;
    let val: i32 = text.trim().parse().ok()?;
    if val < 0 {
        None
    } else {
        Some(val as u32)
    }
}

/// Scan /sys/bus/pci/devices/ for the first GPU-class device and return its NUMA node.
///
/// GPU PCI class codes: 0x030200 (3D controller) and 0x030000 (VGA compatible).
/// Reads the `class` sysfs file for each device and matches the prefix.
#[cfg(all(feature = "numa", target_os = "linux"))]
fn scan_gpu_numa_node() -> Option<u32> {
    let dir = std::fs::read_dir("/sys/bus/pci/devices/").ok()?;
    let mut entries: Vec<_> = dir.flatten().collect();
    // Sort for deterministic behaviour across runs.
    entries.sort_by_key(|e| e.path());
    for entry in entries {
        let base = entry.path();
        let class_text =
            std::fs::read_to_string(base.join("class")).unwrap_or_default();
        let class = class_text.trim().trim_start_matches("0x");
        // 0302xx = 3D controller, 0300xx = VGA / display controller
        if class.starts_with("0302") || class.starts_with("0300") {
            if let Ok(text) = std::fs::read_to_string(base.join("numa_node")) {
                let val: i32 = text.trim().parse().unwrap_or(-1);
                if val >= 0 {
                    return Some(val as u32);
                }
            }
        }
    }
    None
}
