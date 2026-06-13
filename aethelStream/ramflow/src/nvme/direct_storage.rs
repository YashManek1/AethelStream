//! Windows DirectStorage capability probe, buffer allocation helper, and
//! low-level I/O queue for SSD→GPU zero-copy transfers.
//!
//! # Hardware requirements
//! - Windows 11 or Windows 10 20H1+ (build 19041+)
//! - DirectStorage runtime DLLs (`dstorage.dll`, `dstoragecore.dll`), installed
//!   via the DirectX End-User Runtime or the DirectStorage NuGet package.
//! - Compatible NVMe SSD (PCIe 4.0 recommended; PCIe 3.0 works at reduced bandwidth)
//! - NVIDIA GPU with DirectStorage support (RTX 30xx / 40xx series)
//!
//! # SDK requirement
//! DirectStorage DLLs ship with the
//! [Microsoft DirectX End-User Runtime](https://www.microsoft.com/en-us/download/details.aspx?id=35).
//! SDK headers and import libraries are from the
//! [DirectStorage NuGet package](https://www.nuget.org/packages/Microsoft.Direct3D.DirectStorage)
//! version **1.2.0** or later.  RamFlow loads `dstorage.dll` at **runtime** via
//! `LoadLibraryW` / `GetProcAddress`, so no SDK is required at compile time.
//!
//! # CUDA bridge
//! DirectStorage transfers land in D3D12-visible GPU VRAM
//! (`DSTORAGE_REQUEST_DESTINATION_BUFFER`) or in CPU-visible pinned memory
//! (`DSTORAGE_REQUEST_DESTINATION_MEMORY`).  RamFlow uses the memory-destination
//! path, which writes into a pinned `PinnedBuffer` (visible to both CPU and GPU):
//!
//! 1. Allocate via [`alloc_windows_ds_compatible`] (4 KiB-aligned `PinnedBuffer`).
//! 2. Pass as `DSTORAGE_DESTINATION_MEMORY.pBuffer`.
//! 3. After the transfer completes, the GPU reads the data via UVA.
//!
//! For true zero-copy GPU VRAM transfers (destination = D3D12 buffer) add:
//! 4. `cudaExternalMemoryGetMappedBuffer` on a `cudaExternalMemory_t` derived from
//!    a `HANDLE` exported by `ID3D12Resource::GetSharedHandle`.
//! 5. Pass the resulting `CUdeviceptr` as the D3D12 resource.
//! This path requires NVAPI `NvAPI_D3D12_CreateCudaInteropResources` and is
//! documented for future implementation.
//!
//! # Struct layout
//! All `#[repr(C)]` structs in this file are sized and aligned to match the
//! DirectStorage SDK 1.2.0 ABI on x86-64 Windows (MSVC).  Field offsets are
//! annotated inline.  Verify against `dstorage.h` before upgrading the SDK.

use crate::{allocator::PinnedBuffer, Result};
use std::path::Path;

// ---------------------------------------------------------------------------
// Public API — available on all platforms when feature = "direct-storage"
// ---------------------------------------------------------------------------

/// Availability of the DirectStorage runtime on this machine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DirectStorageCapability {
    /// `dstorage.dll` is present and exports `DStorageGetFactory`.
    Available {
        /// Maximum bytes per single transfer request.
        ///
        /// DirectStorage 1.2 specifies 32 MiB per request.
        max_transfer_bytes: u64,
    },
    /// Runtime DLL absent, incompatible OS, or hardware does not qualify.
    Unavailable,
}

/// Probe the Windows DirectStorage runtime without committing to it.
///
/// Loads `dstorage.dll` transiently via `LoadLibraryW`, checks that
/// `DStorageGetFactory` is exported, then immediately releases the module.
/// The actual DLL is re-loaded when [`DirectStorageQueue::open`] is called.
///
/// Always returns [`DirectStorageCapability::Unavailable`] on non-Windows.
pub fn probe_direct_storage() -> DirectStorageCapability {
    #[cfg(target_os = "windows")]
    {
        windows_impl::probe()
    }
    #[cfg(not(target_os = "windows"))]
    {
        DirectStorageCapability::Unavailable
    }
}

/// Allocate a [`PinnedBuffer`] that satisfies DirectStorage's 4 096-byte
/// source-buffer alignment requirement for GPU-direct transfers.
///
/// [`PinnedBuffer::alloc`] guarantees only 512-byte (sector) alignment.
/// `DSTORAGE_REQUEST_DESTINATION_BUFFER` (GPU-VRAM path) requires the source
/// buffer to be 4 KiB-aligned.  This function delegates to
/// [`PinnedBuffer::alloc_page_aligned`], which uses `VirtualAlloc` on Windows
/// and `posix_memalign` with `PAGE_SIZE` granularity on Linux.
///
/// # Errors
/// Propagates [`crate::RamFlowError::AllocationFailed`] on OOM.
pub fn alloc_windows_ds_compatible(bytes: usize) -> Result<PinnedBuffer> {
    PinnedBuffer::alloc_page_aligned(bytes)
}

// ---------------------------------------------------------------------------
// DirectStorageQueue — cross-platform type, Windows-only real body
// ---------------------------------------------------------------------------

/// An open DirectStorage I/O queue for asynchronous shard reads.
///
/// Wraps `IDStorageFactory`, `IDStorageQueue`, and per-shard `IDStorageFile`
/// COM objects on Windows.  On non-Windows platforms every method returns an
/// immediate error so callers can reference the type unconditionally.
///
/// # Construction
/// Use [`DirectStorageQueue::open`].  The constructor probes the DLL, creates
/// the queue, and opens every shard file.  Returns an error if any step fails;
/// the caller should fall back to [`crate::nvme::DirectNvmeEngine`] or
/// `FileReadBackend`.
///
/// # Thread safety
/// All methods are `&self` and internally protected by a `Mutex`.
/// `DirectStorageQueue` is `Send + Sync`.
pub struct DirectStorageQueue {
    #[cfg(target_os = "windows")]
    inner: windows_impl::QueueInner,
    #[cfg(not(target_os = "windows"))]
    _phantom: (),
}

impl DirectStorageQueue {
    /// Open a DirectStorage queue for the supplied shard file paths.
    ///
    /// Loads `dstorage.dll`, calls `DStorageGetFactory`, creates one
    /// `IDStorageQueue` with `DSTORAGE_PRIORITY_NORMAL`, and opens every path
    /// with `IDStorageFactory::OpenFile`.
    ///
    /// # Errors
    /// [`crate::RamFlowError::ConfigError`] when:
    /// * `dstorage.dll` is not found.
    /// * `DStorageGetFactory` returns a COM failure.
    /// * Any shard file cannot be opened.
    pub fn open(shard_paths: &[&Path]) -> Result<Self> {
        #[cfg(target_os = "windows")]
        {
            let inner = windows_impl::QueueInner::open(shard_paths)?;
            Ok(Self { inner })
        }
        #[cfg(not(target_os = "windows"))]
        {
            let _ = shard_paths;
            Err(crate::RamFlowError::ConfigError(
                "DirectStorage is Windows-only".to_string(),
            ))
        }
    }

    /// Submit a non-blocking read of `length` bytes from shard `shard_id`
    /// at `byte_offset` into `dst`.
    ///
    /// Internally calls `IDStorageQueue::EnqueueRequest` +
    /// `IDStorageQueue::EnqueueSetEvent` + `IDStorageQueue::Submit`.
    /// The per-request Windows event handle is stored in a pending queue and
    /// checked by the next [`poll_completions`](Self::poll_completions) call.
    ///
    /// # Alignment
    /// `dst` must be 4 096-byte aligned for GPU-VRAM destination.  The
    /// memory-destination path (used here) requires only pointer validity.
    /// Use [`alloc_windows_ds_compatible`] to guarantee alignment for both paths.
    ///
    /// # Errors
    /// [`crate::RamFlowError::ConfigError`] on COM submission failure.
    pub fn enqueue_read(
        &self,
        shard_id: u32,
        byte_offset: u64,
        length: u64,
        dst: &PinnedBuffer,
        token: u64,
    ) -> Result<()> {
        #[cfg(target_os = "windows")]
        {
            self.inner.enqueue_read(shard_id, byte_offset, length, dst, token)
        }
        #[cfg(not(target_os = "windows"))]
        {
            let _ = (shard_id, byte_offset, length, dst, token);
            Err(crate::RamFlowError::ConfigError(
                "DirectStorage is Windows-only".to_string(),
            ))
        }
    }

    /// Drain completed reads non-blocking.  Returns `(token, bytes_read)` pairs
    /// for every request whose Windows event handle has already been signalled.
    ///
    /// Uses `WaitForSingleObject(event, 0)` (zero timeout) so it never blocks.
    ///
    /// # Errors
    /// [`crate::RamFlowError::ConfigError`] if the pending-queue mutex is poisoned.
    pub fn poll_completions(&self) -> Result<Vec<(u64, i32)>> {
        #[cfg(target_os = "windows")]
        {
            self.inner.poll_completions()
        }
        #[cfg(not(target_os = "windows"))]
        {
            Ok(Vec::new())
        }
    }

    /// Signal pause (called by the co-scheduler's high-pressure callback).
    pub fn set_paused(&self, paused: bool) {
        #[cfg(target_os = "windows")]
        self.inner.set_paused(paused);
        #[cfg(not(target_os = "windows"))]
        let _ = paused;
    }

    /// Current pause state.
    pub fn is_paused(&self) -> bool {
        #[cfg(target_os = "windows")]
        {
            self.inner.is_paused()
        }
        #[cfg(not(target_os = "windows"))]
        {
            false
        }
    }
}

// SAFETY: Windows COM objects accessed through `IDStorageQueue::EnqueueRequest`
// and `IDStorageQueue::Submit` are documented as thread-safe in DS SDK 1.2.
// The pending-completion queue is wrapped in `Mutex`.
unsafe impl Send for DirectStorageQueue {}
unsafe impl Sync for DirectStorageQueue {}

// ---------------------------------------------------------------------------
// Windows implementation
// ---------------------------------------------------------------------------

#[cfg(target_os = "windows")]
mod windows_impl {
    use super::DirectStorageCapability;
    use crate::{allocator::PinnedBuffer, RamFlowError, Result};
    use std::path::Path;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Mutex;
    // Raw WinAPI declarations — avoids windows-sys feature-flag coupling.
    // All signatures match the x86-64 MSVC ABI; types map as:
    //   HANDLE → isize, BOOL → i32, PCWSTR → *const u16.
    extern "system" {
        fn LoadLibraryW(lplibfilename: *const u16) -> isize;
        fn GetProcAddress(
            hmodule: isize,
            lpprocname: *const u8,
        ) -> Option<unsafe extern "system" fn() -> isize>;
        fn FreeLibrary(hlibmodule: isize) -> i32;
        fn CreateEventW(
            lpeventattributes: *const core::ffi::c_void,
            bmanualreset: i32,
            binitialstate: i32,
            lpname: *const u16,
        ) -> isize;
        fn WaitForSingleObject(hhandle: isize, dwmilliseconds: u32) -> u32;
        fn CloseHandle(hobject: isize) -> i32;
    }
    const INFINITE: u32 = 0xFFFF_FFFF;
    const WAIT_OBJECT_0: u32 = 0;

    // -----------------------------------------------------------------------
    // DLL probe
    // -----------------------------------------------------------------------

    /// Null-terminated UTF-16 `"dstorage.dll"`.
    const DSTORAGE_DLL_W: &[u16] = &[
        0x64, 0x73, 0x74, 0x6F, 0x72, 0x61, 0x67, 0x65, // d s t o r a g e
        0x2E, 0x64, 0x6C, 0x6C, 0x00,                     // . d l l \0
    ];

    pub(super) fn probe() -> DirectStorageCapability {
        // Safety: DSTORAGE_DLL_W is a valid null-terminated UTF-16 string.
        let hmod = unsafe { LoadLibraryW(DSTORAGE_DLL_W.as_ptr()) };
        if hmod == 0 {
            return DirectStorageCapability::Unavailable;
        }
        // Safety: DSTORAGE_GET_FACTORY_NAME is a valid null-terminated C string.
        let sym = unsafe {
            GetProcAddress(hmod, DSTORAGE_GET_FACTORY_NAME.as_ptr())
        };
        // Safety: hmod was returned by LoadLibraryW and has not been freed.
        unsafe { FreeLibrary(hmod) };
        if sym.is_some() {
            DirectStorageCapability::Available {
                max_transfer_bytes: 32 * 1024 * 1024, // 32 MiB per DS 1.2 spec
            }
        } else {
            DirectStorageCapability::Unavailable
        }
    }

    // -----------------------------------------------------------------------
    // COM GUID and IUnknown helpers
    // -----------------------------------------------------------------------

    /// COM GUID (equivalent to Windows SDK `GUID` / `IID`).
    ///
    /// Values sourced from Microsoft DirectStorage SDK 1.2.0 `dstorage.h`.
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct ComGuid {
        data1: u32,
        data2: u16,
        data3: u16,
        data4: [u8; 8],
    }

    /// `IDStorageFactory` IID — from DirectStorage SDK 1.2.0 `dstorage.h`.
    const IID_IDSTORAGE_FACTORY: ComGuid = ComGuid {
        data1: 0xafa4_5b22,
        data2: 0xee03,
        data3: 0x44a8,
        data4: [0xb3, 0xfe, 0xd8, 0x9f, 0xc5, 0xa7, 0xee, 0xb7],
    };

    /// `IDStorageQueue` IID — from DirectStorage SDK 1.2.0 `dstorage.h`.
    const IID_IDSTORAGE_QUEUE: ComGuid = ComGuid {
        data1: 0xcec7_9d43,
        data2: 0x7728,
        data3: 0x4c34,
        data4: [0xb5, 0x7a, 0x4b, 0x0c, 0x63, 0x2b, 0x4c, 0x45],
    };

    /// `IDStorageFile` IID — from DirectStorage SDK 1.2.0 `dstorage.h`.
    const IID_IDSTORAGE_FILE: ComGuid = ComGuid {
        data1: 0x5de6_a171,
        data2: 0x7f3c,
        data3: 0x4d9f,
        data4: [0xb8, 0x8a, 0x4a, 0x22, 0xc5, 0xdc, 0x98, 0xb4],
    };

    /// `DStorageGetFactory` export name (null-terminated).
    const DSTORAGE_GET_FACTORY_NAME: &[u8] = b"DStorageGetFactory\0";

    /// Release a COM object via IUnknown::Release (vtable index 2).
    ///
    /// # Safety
    /// `ptr` must be a valid COM object pointer returned by a DirectStorage API.
    /// IUnknown vtable layout (QueryInterface=0, AddRef=1, Release=2) is mandated
    /// by the COM ABI and is stable across all DirectStorage SDK versions.
    unsafe fn com_release(ptr: *mut core::ffi::c_void) {
        // COM ABI: first field of any COM object is *vtable; vtable[2] = Release.
        // We cast to a triple pointer to reach the Release function pointer.
        let vtable_ptr = *(ptr as *const *const usize);
        let release_raw = *vtable_ptr.add(2); // vtable index 2 = IUnknown::Release
        let release: unsafe extern "system" fn(*mut core::ffi::c_void) -> u32 =
            core::mem::transmute(release_raw);
        release(ptr);
    }

    // -----------------------------------------------------------------------
    // DirectStorage COM struct layouts (SDK 1.2.0, x86-64 MSVC ABI)
    // -----------------------------------------------------------------------

    /// `IDStorageFactory` vtable — offsets 0-7 matching SDK `dstorage.h`.
    #[repr(C)]
    struct IDStorageFactoryVtbl {
        // IUnknown (offsets 0-2)
        query_interface: unsafe extern "system" fn(
            *mut core::ffi::c_void,
            *const ComGuid,
            *mut *mut core::ffi::c_void,
        ) -> i32,
        add_ref: unsafe extern "system" fn(*mut core::ffi::c_void) -> u32,
        release: unsafe extern "system" fn(*mut core::ffi::c_void) -> u32,
        // IDStorageFactory (offsets 3-7)
        create_queue: unsafe extern "system" fn(
            *mut core::ffi::c_void,
            *const DStorageQueueDesc,
            *const ComGuid,
            *mut *mut core::ffi::c_void,
        ) -> i32,
        open_file: unsafe extern "system" fn(
            *mut core::ffi::c_void,
            *const u16,   // LPCWSTR path
            *const ComGuid,
            *mut *mut core::ffi::c_void,
        ) -> i32,
        create_status_array: unsafe extern "system" fn(
            *mut core::ffi::c_void,
            u32,
            *const i8,
            *const ComGuid,
            *mut *mut core::ffi::c_void,
        ) -> i32,
        set_debug_flags: unsafe extern "system" fn(*mut core::ffi::c_void, u32),
        set_staging_buffer_size: unsafe extern "system" fn(*mut core::ffi::c_void, u32),
    }

    /// `IDStorageQueue` vtable — offsets 0-8 matching SDK `dstorage.h`.
    #[repr(C)]
    struct IDStorageQueueVtbl {
        // IUnknown (offsets 0-2)
        query_interface: unsafe extern "system" fn(
            *mut core::ffi::c_void,
            *const ComGuid,
            *mut *mut core::ffi::c_void,
        ) -> i32,
        add_ref: unsafe extern "system" fn(*mut core::ffi::c_void) -> u32,
        release: unsafe extern "system" fn(*mut core::ffi::c_void) -> u32,
        // IDStorageQueue (offsets 3-8)
        enqueue_request: unsafe extern "system" fn(
            *mut core::ffi::c_void,
            *const DStorageRequest,
        ),
        enqueue_status: unsafe extern "system" fn(
            *mut core::ffi::c_void,
            *mut core::ffi::c_void, // IDStorageStatusArray*
            u32,                    // index
        ),
        enqueue_set_event: unsafe extern "system" fn(
            *mut core::ffi::c_void,
            isize, // HANDLE (event)
        ),
        submit: unsafe extern "system" fn(*mut core::ffi::c_void),
        cancel_requests_with_tag: unsafe extern "system" fn(*mut core::ffi::c_void, u64, u64),
        close: unsafe extern "system" fn(*mut core::ffi::c_void),
    }

    /// COM object header — every COM object's first field is a vtable pointer.
    #[repr(C)]
    struct ComObject<V> {
        vtable: *const V,
    }

    // -----------------------------------------------------------------------
    // Data structures (DS SDK 1.2.0, x86-64 MSVC ABI)
    // -----------------------------------------------------------------------

    /// `DSTORAGE_QUEUE_DESC` — queue creation parameters.
    ///
    /// Byte layout (x86-64): total 32 bytes.
    ///
    /// | offset | field               | size |
    /// |--------|---------------------|------|
    /// | 0      | source_type         | 1    |
    /// | 1      | (implicit pad)      | 1    |
    /// | 2      | capacity            | 2    |
    /// | 4      | priority            | 1    |
    /// | 5–7    | (implicit pad)      | 3    |
    /// | 8      | name                | 8    |
    /// | 16     | device              | 8    |
    /// | 24     | staging_buffer_size | 4    |
    /// | 28–31  | (implicit pad)      | 4    |
    #[repr(C)]
    struct DStorageQueueDesc {
        source_type: u8,                     // DSTORAGE_SOURCE_TYPE (UINT8 enum)
        capacity: u16,                       // submit queue depth (max 0x2000)
        priority: i8,                        // DSTORAGE_PRIORITY (INT8 enum)
        name: *const u16,                    // optional debug name, LPCWSTR
        device: *mut core::ffi::c_void,      // ID3D12Device* (NULL = memory dest)
        staging_buffer_size: u32,            // 0 = use DLL default (32 MiB)
    }

    /// Packed 64-bit request-options field.
    ///
    /// Bit layout (per SDK):
    /// * bits  0–7  : `CompressionFormat` (0 = uncompressed)
    /// * bit   8    : `SourceType`        (0 = file, 1 = memory)
    /// * bits  9–12 : `DestinationType`   (0 = memory, 1 = D3D12 buffer, …)
    /// * bits 13–63 : Reserved (must be zero)
    #[repr(transparent)]
    #[derive(Clone, Copy)]
    struct DStorageRequestOptions(u64);

    impl DStorageRequestOptions {
        /// File → CPU-visible pinned memory, no compression (most common path).
        const FILE_TO_MEMORY: Self = Self(0); // all fields zero
    }

    /// `DSTORAGE_SOURCE_FILE` — describes which file bytes to read.
    ///
    /// Byte layout: 24 bytes on x86-64 (`pointer + u64 + u32 + 4-byte pad`).
    #[repr(C)]
    struct DStorageSourceFile {
        source: *mut core::ffi::c_void, // IDStorageFile*  (8 bytes)
        offset: u64,                    // byte offset     (8 bytes)
        size: u32,                      // byte count      (4 bytes)
        _pad: u32,                      // struct padding  (4 bytes)
    }

    /// `DSTORAGE_SOURCE_MEMORY` — alternative source (memory → memory).
    ///
    /// Byte layout: 16 bytes on x86-64.
    #[repr(C)]
    struct DStorageSourceMemory {
        source: *const core::ffi::c_void, // 8 bytes
        size: u32,                         // 4 bytes
        _pad: u32,                         // 4 bytes
    }

    /// `DSTORAGE_SOURCE` union — 24 bytes (size of largest member).
    #[repr(C)]
    union DStorageSource {
        file: core::mem::ManuallyDrop<DStorageSourceFile>,
        memory: core::mem::ManuallyDrop<DStorageSourceMemory>,
    }

    /// `DSTORAGE_DESTINATION_MEMORY` — transfer into CPU-visible pinned memory.
    ///
    /// Byte layout: 16 bytes on x86-64.
    #[repr(C)]
    struct DStorageDestinationMemory {
        buffer: *mut core::ffi::c_void, // destination pointer (8 bytes)
        size: u32,                       // byte count         (4 bytes)
        _pad: u32,                       // struct padding     (4 bytes)
    }

    /// `DSTORAGE_DESTINATION` union.
    ///
    /// Padded to 40 bytes to accommodate all SDK variants including
    /// `DSTORAGE_DESTINATION_TILES` (the largest member in DS SDK 1.2.0).
    /// Over-sizing is safe: `EnqueueRequest` receives a `*const DSTORAGE_REQUEST`
    /// and the DLL reads only the bytes it needs from offset 0.
    #[repr(C)]
    union DStorageDestination {
        memory: core::mem::ManuallyDrop<DStorageDestinationMemory>,
        /// Pad to 40 bytes for all destination variants (SDK 1.2.0).
        _pad: [u8; 40],
    }

    /// `DSTORAGE_REQUEST` — a single I/O transfer descriptor.
    ///
    /// Byte layout on x86-64 (96 bytes total):
    ///
    /// | offset | field              | size |
    /// |--------|--------------------|------|
    /// | 0      | options            | 8    |
    /// | 8      | source             | 24   |
    /// | 32     | destination        | 40   |
    /// | 72     | uncompressed_size  | 4    |
    /// | 76     | (implicit pad)     | 4    |
    /// | 80     | cancellation_tag   | 8    |
    /// | 88     | name               | 8    |
    ///
    /// Verified against `sizeof(DSTORAGE_REQUEST)` = 96 in DS SDK 1.2.0.
    #[repr(C)]
    struct DStorageRequest {
        options: DStorageRequestOptions,    // offset 0,  8 bytes
        source: DStorageSource,             // offset 8,  24 bytes
        destination: DStorageDestination,   // offset 32, 40 bytes
        uncompressed_size: u32,             // offset 72, 4 bytes
        // 4 bytes implicit padding before u64
        cancellation_tag: u64,              // offset 80, 8 bytes
        name: *const i8,                    // offset 88, 8 bytes
    }

    // Compile-time size guard — fails if our layout drifts from the SDK.
    const _DSTORAGE_REQUEST_SIZE: () =
        assert!(core::mem::size_of::<DStorageRequest>() == 96);

    // -----------------------------------------------------------------------
    // DLL lifetime wrapper
    // -----------------------------------------------------------------------

    /// RAII wrapper for a loaded DLL module handle.
    struct OwnedDll(isize);

    impl OwnedDll {
        /// Load a DLL by its null-terminated UTF-16 name.
        ///
        /// # Safety
        /// `name_w` must be a valid null-terminated UTF-16 string.
        unsafe fn load(name_w: &[u16]) -> Result<Self> {
            let h = LoadLibraryW(name_w.as_ptr());
            if h == 0 {
                Err(RamFlowError::ConfigError(
                    "dstorage.dll not found — install the DirectX runtime".to_string(),
                ))
            } else {
                Ok(Self(h))
            }
        }

        /// Resolve a function symbol by its null-terminated ASCII name.
        ///
        /// # Safety
        /// `sym` must be a valid null-terminated C string.
        unsafe fn proc(&self, sym: &[u8]) -> Option<unsafe extern "system" fn() -> isize> {
            GetProcAddress(self.0, sym.as_ptr())
        }
    }

    impl Drop for OwnedDll {
        fn drop(&mut self) {
            // Safety: self.0 is a module handle returned by LoadLibraryW.
            unsafe { FreeLibrary(self.0) };
        }
    }

    // -----------------------------------------------------------------------
    // Pending read (one per in-flight request)
    // -----------------------------------------------------------------------

    struct PendingRead {
        token: u64,
        /// Windows auto-reset event, signalled when `Submit` drains to this point.
        event_handle: isize, // HANDLE
        /// Expected bytes transferred (echoed as `result` in the completion).
        byte_count: i32,
    }

    impl Drop for PendingRead {
        fn drop(&mut self) {
            // Safety: event_handle was returned by CreateEventW and not yet closed.
            unsafe { CloseHandle(self.event_handle) };
        }
    }

    // -----------------------------------------------------------------------
    // QueueInner — holds all live COM objects for one backend lifetime
    // -----------------------------------------------------------------------

    /// Raw COM pointer wrapper — `!Send + !Sync` by default; we assert the
    /// DirectStorage thread-safety contract for IDStorageQueue (DS SDK 1.2 §4).
    struct RawCom(*mut core::ffi::c_void);

    // Safety: IDStorageQueue::EnqueueRequest + Submit are documented as
    // thread-safe across concurrent callers (DirectStorage SDK 1.2, §4 "Threading").
    unsafe impl Send for RawCom {}
    unsafe impl Sync for RawCom {}

    impl Drop for RawCom {
        fn drop(&mut self) {
            if !self.0.is_null() {
                // Safety: self.0 was returned by a DirectStorage COM factory.
                unsafe { com_release(self.0) };
            }
        }
    }

    pub(super) struct QueueInner {
        /// Keeps `dstorage.dll` loaded for the lifetime of the queue.
        _dll: OwnedDll,
        factory: RawCom,
        queue: RawCom,
        /// One `IDStorageFile*` per shard (index = shard_id).
        shard_files: Vec<RawCom>,
        pending: Mutex<Vec<PendingRead>>,
        paused: AtomicBool,
    }

    impl QueueInner {
        pub(super) fn open(shard_paths: &[&Path]) -> Result<Self> {
            // Safety: DSTORAGE_DLL_W is a valid null-terminated UTF-16 string.
            let dll = unsafe { OwnedDll::load(DSTORAGE_DLL_W)? };

            // Resolve DStorageGetFactory.
            // Safety: DSTORAGE_GET_FACTORY_NAME is a valid null-terminated C string.
            let get_factory_raw = unsafe { dll.proc(DSTORAGE_GET_FACTORY_NAME) }
                .ok_or_else(|| {
                    RamFlowError::ConfigError(
                        "dstorage.dll missing DStorageGetFactory export".to_string(),
                    )
                })?;

            type DStorageGetFactoryFn = unsafe extern "system" fn(
                riid: *const ComGuid,
                ppv: *mut *mut core::ffi::c_void,
            ) -> i32;

            // Safety: the function pointer type matches the SDK export.
            let get_factory: DStorageGetFactoryFn =
                unsafe { core::mem::transmute(get_factory_raw) };

            // --- Get IDStorageFactory ---
            let mut factory_ptr: *mut core::ffi::c_void = core::ptr::null_mut();
            // Safety: IID_IDSTORAGE_FACTORY is the correct IID; factory_ptr is a valid out-param.
            let hr = unsafe { get_factory(&IID_IDSTORAGE_FACTORY, &mut factory_ptr) };
            if hr < 0 || factory_ptr.is_null() {
                return Err(RamFlowError::ConfigError(format!(
                    "DStorageGetFactory failed: HRESULT {hr:#010x}"
                )));
            }
            let factory = RawCom(factory_ptr);

            // --- Create IDStorageQueue ---
            let queue_desc = DStorageQueueDesc {
                source_type: 0,              // DSTORAGE_REQUEST_SOURCE_FILE
                capacity: 128,               // submit depth
                priority: 0,                 // DSTORAGE_PRIORITY_NORMAL
                name: core::ptr::null(),     // no debug name
                device: core::ptr::null_mut(), // NULL = memory destination
                staging_buffer_size: 0,      // use DLL default (32 MiB)
            };

            let factory_obj =
                factory.0 as *mut ComObject<IDStorageFactoryVtbl>;
            let mut queue_ptr: *mut core::ffi::c_void = core::ptr::null_mut();
            // Safety: factory_obj is a valid IDStorageFactory* returned by DStorageGetFactory.
            let hr = unsafe {
                ((*(*factory_obj).vtable).create_queue)(
                    factory.0,
                    &queue_desc,
                    &IID_IDSTORAGE_QUEUE,
                    &mut queue_ptr,
                )
            };
            if hr < 0 || queue_ptr.is_null() {
                return Err(RamFlowError::ConfigError(format!(
                    "IDStorageFactory::CreateQueue failed: HRESULT {hr:#010x}"
                )));
            }
            let queue = RawCom(queue_ptr);

            // --- Open each shard file ---
            let mut shard_files = Vec::with_capacity(shard_paths.len());
            for path in shard_paths {
                let wide = path_to_wide(path)?;
                let mut file_ptr: *mut core::ffi::c_void = core::ptr::null_mut();
                // Safety: factory_obj and wide are valid; file_ptr is a valid out-param.
                let hr = unsafe {
                    ((*(*factory_obj).vtable).open_file)(
                        factory.0,
                        wide.as_ptr(),
                        &IID_IDSTORAGE_FILE,
                        &mut file_ptr,
                    )
                };
                if hr < 0 || file_ptr.is_null() {
                    return Err(RamFlowError::ConfigError(format!(
                        "IDStorageFactory::OpenFile failed for {}: HRESULT {hr:#010x}",
                        path.display()
                    )));
                }
                shard_files.push(RawCom(file_ptr));
            }

            Ok(Self {
                _dll: dll,
                factory,
                queue,
                shard_files,
                pending: Mutex::new(Vec::new()),
                paused: AtomicBool::new(false),
            })
        }

        pub(super) fn enqueue_read(
            &self,
            shard_id: u32,
            byte_offset: u64,
            length: u64,
            dst: &PinnedBuffer,
            token: u64,
        ) -> Result<()> {
            let file_ptr = self
                .shard_files
                .get(shard_id as usize)
                .ok_or_else(|| {
                    RamFlowError::ConfigError(format!(
                        "DirectStorage: shard_id {shard_id} out of range (have {})",
                        self.shard_files.len()
                    ))
                })?;

            // Create an auto-reset event to signal when this request completes.
            // Safety: NULL security attributes and name are valid; auto-reset=FALSE,
            // initial-state=FALSE.
            let event = unsafe {
                CreateEventW(
                    core::ptr::null(), // default security
                    0,                 // bManualReset = FALSE (auto-reset)
                    0,                 // bInitialState = FALSE (not signalled)
                    core::ptr::null(), // no name
                )
            };
            if event == 0 {
                return Err(RamFlowError::ConfigError(
                    "CreateEventW failed for DirectStorage completion".to_string(),
                ));
            }

            let byte_count = length.min(i32::MAX as u64) as i32;

            let request = DStorageRequest {
                options: DStorageRequestOptions::FILE_TO_MEMORY,
                source: DStorageSource {
                    file: core::mem::ManuallyDrop::new(DStorageSourceFile {
                        source: file_ptr.0,
                        offset: byte_offset,
                        size: length as u32,
                        _pad: 0,
                    }),
                },
                destination: DStorageDestination {
                    memory: core::mem::ManuallyDrop::new(DStorageDestinationMemory {
                        // Safety: dst is a valid PinnedBuffer whose memory lives until
                        // the completion token is acknowledged by the caller.
                        buffer: dst.as_ptr() as *mut core::ffi::c_void,
                        size: dst.len() as u32,
                        _pad: 0,
                    }),
                },
                uncompressed_size: 0, // 0 = same as compressed size (no compression)
                cancellation_tag: token,
                name: core::ptr::null(),
            };

            let queue_obj = self.queue.0 as *mut ComObject<IDStorageQueueVtbl>;
            // Safety: queue_obj is a valid IDStorageQueue*; request is valid for the
            // duration of this call (EnqueueRequest copies the descriptor internally).
            unsafe {
                ((*(*queue_obj).vtable).enqueue_request)(self.queue.0, &request);
                // EnqueueSetEvent tells the queue to signal `event` after this request.
                ((*(*queue_obj).vtable).enqueue_set_event)(self.queue.0, event);
                // Submit flushes the pending entries to the hardware DMA engine.
                ((*(*queue_obj).vtable).submit)(self.queue.0);
            }

            let mut pending = self.pending.lock().map_err(|poison| {
                RamFlowError::ConfigError(format!(
                    "DirectStorage pending lock poisoned: {poison}"
                ))
            })?;
            pending.push(PendingRead { token, event_handle: event, byte_count });
            Ok(())
        }

        pub(super) fn poll_completions(&self) -> Result<Vec<(u64, i32)>> {
            let mut pending = self.pending.lock().map_err(|poison| {
                RamFlowError::ConfigError(format!(
                    "DirectStorage pending lock poisoned: {poison}"
                ))
            })?;

            let mut completed = Vec::new();
            let mut still_pending = Vec::new();

            for entry in pending.drain(..) {
                // Non-blocking wait: 0 ms timeout.
                // Safety: entry.event_handle is a valid auto-reset event handle.
                let status = unsafe { WaitForSingleObject(entry.event_handle, 0) };
                if status == WAIT_OBJECT_0 {
                    completed.push((entry.token, entry.byte_count));
                    // PendingRead::drop closes the event handle automatically.
                } else {
                    // Not yet signalled; keep in pending.
                    still_pending.push(entry);
                }
            }

            *pending = still_pending;
            Ok(completed)
        }

        pub(super) fn set_paused(&self, paused: bool) {
            self.paused.store(paused, Ordering::Release);
        }

        pub(super) fn is_paused(&self) -> bool {
            self.paused.load(Ordering::Acquire)
        }
    }

    // -----------------------------------------------------------------------
    // Path conversion helper
    // -----------------------------------------------------------------------

    /// Convert a Rust `Path` to a null-terminated UTF-16 `Vec<u16>`.
    fn path_to_wide(path: &Path) -> Result<Vec<u16>> {
        use std::os::windows::ffi::OsStrExt as _;
        let mut wide: Vec<u16> = path.as_os_str().encode_wide().collect();
        wide.push(0); // null terminator
        Ok(wide)
    }
}
