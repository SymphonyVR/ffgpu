//! Dynamic CUDA Driver API bindings for WSL2 / Linux zero-copy video decode.
//!
//! Provides direct interop between Vulkan textures (exported via VK_KHR_external_memory_fd)
//! and NVIDIA CUDA/NVDEC hardware video decode surfaces without host-memory copies.

#![allow(non_camel_case_types, non_snake_case, dead_code)]

#[cfg(target_os = "linux")]
use std::ffi::c_char;
use std::ffi::c_void;
use std::sync::OnceLock;

pub type CUdevice = i32;
pub type CUcontext = *mut c_void;
pub type CUstream = *mut c_void;
pub type CUarray = *mut c_void;
pub type CUmipmappedArray = *mut c_void;
pub type CUexternalMemory = *mut c_void;
pub type CUdeviceptr = u64;

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CUresult(pub i32);

pub const CUDA_SUCCESS: CUresult = CUresult(0);
pub const CUDA_EXTERNAL_MEMORY_DEDICATED: u32 = 0x1;

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CUmemorytype {
    Host = 0x01,
    Device = 0x02,
    Array = 0x03,
    Unified = 0x04,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CUexternalMemoryHandleType {
    OpaqueFd = 1,
    OpaqueWin32 = 2,
    OpaqueWin32Kmt = 3,
    D3D12Heap = 4,
    D3D12Resource = 5,
    D3D11Resource = 6,
    D3D11ResourceKmt = 7,
    NvSciBuf = 8,
}

#[repr(C)]
pub union CUexternalMemoryHandle {
    pub fd: i32,
    pub win32: CUexternalMemoryHandleWin32,
    pub nv_sci_buf_object: *const c_void,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CUexternalMemoryHandleWin32 {
    pub handle: *mut c_void,
    pub name: *const c_void,
}

#[repr(C)]
pub struct CUDA_EXTERNAL_MEMORY_HANDLE_DESC {
    pub handle_type: CUexternalMemoryHandleType,
    pub handle: CUexternalMemoryHandle,
    pub size: u64,
    pub flags: u32,
    pub reserved: [u32; 16],
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CUarray_format {
    UnsignedInt8 = 0x01,
    UnsignedInt16 = 0x02,
    UnsignedInt32 = 0x03,
    SignedInt8 = 0x08,
    SignedInt16 = 0x09,
    SignedInt32 = 0x0a,
    Half = 0x10,
    Float = 0x20,
    NV12 = 0xb0,
    UnormInt8x1 = 0xc0,
    UnormInt8x2 = 0xc1,
    UnormInt8x4 = 0xc2,
    UnormInt16x1 = 0xc3,
    UnormInt16x2 = 0xc4,
    UnormInt16x4 = 0xc5,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CUDA_ARRAY3D_DESCRIPTOR {
    pub width: usize,
    pub height: usize,
    pub depth: usize,
    pub format: CUarray_format,
    pub num_channels: u32,
    pub flags: u32,
}

#[repr(C)]
pub struct CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC {
    pub offset: u64,
    pub array_desc: CUDA_ARRAY3D_DESCRIPTOR,
    pub num_levels: u32,
    pub reserved: [u32; 16],
}

#[repr(C)]
pub struct CUDA_MEMCPY2D {
    pub src_x_in_bytes: usize,
    pub src_y: usize,
    pub src_memory_type: CUmemorytype,
    pub src_host: *const c_void,
    pub src_device: CUdeviceptr,
    pub src_array: CUarray,
    pub src_pitch: usize,

    pub dst_x_in_bytes: usize,
    pub dst_y: usize,
    pub dst_memory_type: CUmemorytype,
    pub dst_host: *mut c_void,
    pub dst_device: CUdeviceptr,
    pub dst_array: CUarray,
    pub dst_pitch: usize,

    pub width_in_bytes: usize,
    pub height: usize,
}

#[derive(Clone)]
pub struct CudaDriver {
    pub cu_init: unsafe extern "C" fn(flags: u32) -> CUresult,
    pub cu_driver_get_version: unsafe extern "C" fn(version: *mut i32) -> CUresult,
    pub cu_device_get: unsafe extern "C" fn(device: *mut CUdevice, ordinal: i32) -> CUresult,
    pub cu_device_primary_ctx_retain: unsafe extern "C" fn(pctx: *mut CUcontext, dev: CUdevice) -> CUresult,
    pub cu_ctx_set_current: unsafe extern "C" fn(ctx: CUcontext) -> CUresult,
    pub cu_stream_create: unsafe extern "C" fn(phStream: *mut CUstream, flags: u32) -> CUresult,
    pub cu_stream_destroy: unsafe extern "C" fn(hStream: CUstream) -> CUresult,
    pub cu_stream_synchronize: unsafe extern "C" fn(hStream: CUstream) -> CUresult,
    pub cu_import_external_memory: unsafe extern "C" fn(
        extMem_out: *mut CUexternalMemory,
        memHandleDesc: *const CUDA_EXTERNAL_MEMORY_HANDLE_DESC,
    ) -> CUresult,
    pub cu_destroy_external_memory: unsafe extern "C" fn(extMem: CUexternalMemory) -> CUresult,
    pub cu_external_memory_get_mapped_mipmapped_array: unsafe extern "C" fn(
        mipmappedArray_out: *mut CUmipmappedArray,
        extMem: CUexternalMemory,
        mipmappedArrayDesc: *const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC,
    ) -> CUresult,
    pub cu_mipmapped_array_get_level: unsafe extern "C" fn(
        pLevelArray: *mut CUarray,
        hMipmappedArray: CUmipmappedArray,
        level: u32,
    ) -> CUresult,
    pub cu_mipmapped_array_destroy: unsafe extern "C" fn(hMipmappedArray: CUmipmappedArray) -> CUresult,
    pub cu_memcpy_2d_async: unsafe extern "C" fn(
        pCopy: *const CUDA_MEMCPY2D,
        hStream: CUstream,
    ) -> CUresult,
    pub cu_mem_alloc_host: unsafe extern "C" fn(pp: *mut *mut c_void, bytesize: usize) -> CUresult,
    pub cu_mem_free_host: unsafe extern "C" fn(p: *mut c_void) -> CUresult,
    pub primary_ctx: CUcontext,
}

unsafe impl Send for CudaDriver {}
unsafe impl Sync for CudaDriver {}

static CUDA_DRIVER: OnceLock<Option<CudaDriver>> = OnceLock::new();

#[cfg(target_os = "linux")]
fn load_sym<T>(handle: *mut c_void, name: &str) -> Option<T> {
    let c_str = std::ffi::CString::new(name).ok()?;
    let sym = unsafe { libc::dlsym(handle, c_str.as_ptr()) };
    if sym.is_null() {
        log::warn!("[CUDA] Missing symbol: {}", name);
        None
    } else {
        Some(unsafe { std::mem::transmute_copy(&sym) })
    }
}

pub fn get_cuda_driver() -> Option<&'static CudaDriver> {
    CUDA_DRIVER.get_or_init(|| {
        #[cfg(target_os = "linux")]
        unsafe {
            let candidate_paths = [
                b"/usr/lib/wsl/lib/libcuda.so.1\0".as_ptr() as *const c_char,
                b"libcuda.so.1\0".as_ptr() as *const c_char,
                b"libcuda.so\0".as_ptr() as *const c_char,
            ];

            let mut handle: *mut c_void = std::ptr::null_mut();
            for path in &candidate_paths {
                handle = libc::dlopen(*path, libc::RTLD_LAZY | libc::RTLD_LOCAL);
                if !handle.is_null() {
                    let path_str = std::ffi::CStr::from_ptr(*path).to_string_lossy();
                    log::info!("[CUDA] Loaded CUDA driver from: {}", path_str);
                    break;
                }
            }

            if handle.is_null() {
                log::debug!("[CUDA] libcuda.so not found on this system");
                return None;
            }

            let driver = CudaDriver {
                cu_init: load_sym(handle, "cuInit")?,
                cu_driver_get_version: load_sym(handle, "cuDriverGetVersion")?,
                cu_device_get: load_sym(handle, "cuDeviceGet")?,
                cu_device_primary_ctx_retain: load_sym(handle, "cuDevicePrimaryCtxRetain")?,
                cu_ctx_set_current: load_sym(handle, "cuCtxSetCurrent")?,
                cu_stream_create: load_sym(handle, "cuStreamCreate")?,
                cu_stream_destroy: load_sym(handle, "cuStreamDestroy")?,
                cu_stream_synchronize: load_sym(handle, "cuStreamSynchronize")?,
                cu_import_external_memory: load_sym(handle, "cuImportExternalMemory")?,
                cu_destroy_external_memory: load_sym(handle, "cuDestroyExternalMemory")?,
                cu_external_memory_get_mapped_mipmapped_array: load_sym(
                    handle,
                    "cuExternalMemoryGetMappedMipmappedArray",
                )?,
                cu_mipmapped_array_get_level: load_sym(handle, "cuMipmappedArrayGetLevel")?,
                cu_mipmapped_array_destroy: load_sym(handle, "cuMipmappedArrayDestroy")?,
                cu_memcpy_2d_async: load_sym(handle, "cuMemcpy2DAsync_v2")
                    .or_else(|| load_sym(handle, "cuMemcpy2DAsync"))?,
                cu_mem_alloc_host: load_sym(handle, "cuMemAllocHost_v2")
                    .or_else(|| load_sym(handle, "cuMemAllocHost"))?,
                cu_mem_free_host: load_sym(handle, "cuMemFreeHost")?,
                primary_ctx: std::ptr::null_mut(),
            };

            let res = (driver.cu_init)(0);
            if res != CUDA_SUCCESS {
                log::warn!("[CUDA] cuInit(0) failed with code: {:?}", res);
                return None;
            }

            let mut version = 0i32;
            let _ = (driver.cu_driver_get_version)(&mut version);
            eprintln!("[CUDA] Initialized CUDA driver version: {}", version);

            let mut dev: CUdevice = 0;
            let res = (driver.cu_device_get)(&mut dev, 0);
            if res != CUDA_SUCCESS {
                eprintln!("[CUDA] cuDeviceGet(0) failed: {:?}", res);
                return None;
            }

            let mut primary_ctx: CUcontext = std::ptr::null_mut();
            let res = (driver.cu_device_primary_ctx_retain)(&mut primary_ctx, dev);
            if res != CUDA_SUCCESS || primary_ctx.is_null() {
                eprintln!("[CUDA] cuDevicePrimaryCtxRetain failed: {:?}", res);
                return None;
            }

            let res = (driver.cu_ctx_set_current)(primary_ctx);
            if res != CUDA_SUCCESS {
                eprintln!("[CUDA] cuCtxSetCurrent failed: {:?}", res);
                return None;
            }

            let mut driver = driver;
            driver.primary_ctx = primary_ctx;

            Some(driver)
        }

        #[cfg(not(target_os = "linux"))]
        {
            None
        }
    }).as_ref()
}

pub fn is_cuda_available() -> bool {
    get_cuda_driver().is_some()
}
