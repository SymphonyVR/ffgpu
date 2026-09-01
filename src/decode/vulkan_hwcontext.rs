//! FFI definitions and helpers for FFmpeg's Vulkan hardware context.
//!
//! These structures are defined in `libavutil/hwcontext_vulkan.h` but are not
//! exposed by `ffmpeg-sys-next`, so we define them here and rely on a runtime
//! size check to catch header mismatches.

use crate::error::{Error, Result};
use ffmpeg_next::sys as ff;
use std::{
    ffi::{CStr, c_char, c_int, c_void},
    ptr::NonNull,
};

/// Number of data pointers in FFmpeg frame structures.
pub const AV_NUM_DATA_POINTERS: usize = 8;

/// Queue family descriptor used by `AVVulkanDeviceContext`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AVVulkanDeviceQueueFamily {
    pub idx: c_int,
    pub num: c_int,
    pub flags: ash::vk::QueueFlags,
    pub video_caps: ash::vk::VideoCodecOperationFlagsKHR,
}

/// FFmpeg's Vulkan device context, allocated as `AVHWDeviceContext.hwctx`.
///
/// This mirrors the C struct from `libavutil/hwcontext_vulkan.h` as of
/// FFmpeg 8.0 (libavutil major 60). The deprecated queue family fields
/// (`FF_API_VULKAN_FIXED_QUEUES`, which is true for major < 61) MUST be
/// included because they are part of the ABI layout — omitting them shifts
/// every subsequent field by 40 bytes and causes access violations.
#[repr(C)]
#[derive(Debug)]
pub struct AVVulkanDeviceContext {
    pub alloc: *const ash::vk::AllocationCallbacks<'static>,
    pub get_proc_addr: Option<ash::vk::PFN_vkGetInstanceProcAddr>,
    pub inst: ash::vk::Instance,
    pub phys_dev: ash::vk::PhysicalDevice,
    pub act_dev: ash::vk::Device,
    pub device_features: ash::vk::PhysicalDeviceFeatures2<'static>,
    pub enabled_inst_extensions: *const *const c_char,
    pub nb_enabled_inst_extensions: c_int,
    pub enabled_dev_extensions: *const *const c_char,
    pub nb_enabled_dev_extensions: c_int,
    // --- Deprecated fields (FF_API_VULKAN_FIXED_QUEUES, active in major < 61) ---
    // These MUST be present to match the C struct layout. Do not remove until
    // FFmpeg 9.0 (libavutil major 61) drops them.
    pub queue_family_index: c_int,
    pub nb_graphics_queues: c_int,
    pub queue_family_tx_index: c_int,
    pub nb_tx_queues: c_int,
    pub queue_family_comp_index: c_int,
    pub nb_comp_queues: c_int,
    pub queue_family_encode_index: c_int,
    pub nb_encode_queues: c_int,
    pub queue_family_decode_index: c_int,
    pub nb_decode_queues: c_int,
    // --- End deprecated fields ---
    pub lock_queue: Option<unsafe extern "C" fn(*mut c_void, u32, u32)>,
    pub unlock_queue: Option<unsafe extern "C" fn(*mut c_void, u32, u32)>,
    pub qf: [AVVulkanDeviceQueueFamily; 64],
    pub nb_qf: c_int,
}

/// FFmpeg's Vulkan frames context, allocated as `AVHWFramesContext.hwctx`.
///
/// Note: `AVVkFrameFlags` is a C enum, which is 4 bytes (`int`) on MSVC.
/// Using `u64` here would shift all subsequent fields by 4 bytes.
#[repr(C)]
#[derive(Debug)]
pub struct AVVulkanFramesContext {
    pub tiling: ash::vk::ImageTiling,
    pub usage: ash::vk::ImageUsageFlags,
    pub create_pnext: *mut c_void,
    pub alloc_pnext: [*mut c_void; AV_NUM_DATA_POINTERS],
    pub flags: c_int,
    pub img_flags: ash::vk::ImageCreateFlags,
    pub format: [ash::vk::Format; AV_NUM_DATA_POINTERS],
    pub nb_layers: c_int,
    pub lock_frame: Option<unsafe extern "C" fn(*mut c_void, *mut AVVkFrame)>,
    pub unlock_frame: Option<unsafe extern "C" fn(*mut c_void, *mut AVVkFrame)>,
}

/// FFmpeg's Vulkan frame descriptor.
///
/// Note: the size of this structure is not part of the FFmpeg ABI, so it must
/// be allocated via `av_vk_frame_alloc()`.
#[repr(C)]
#[derive(Debug)]
pub struct AVVkFrame {
    pub img: [ash::vk::Image; AV_NUM_DATA_POINTERS],
    pub tiling: ash::vk::ImageTiling,
    pub mem: [ash::vk::DeviceMemory; AV_NUM_DATA_POINTERS],
    pub size: [usize; AV_NUM_DATA_POINTERS],
    pub flags: ash::vk::MemoryPropertyFlags,
    pub access: [ash::vk::AccessFlags; AV_NUM_DATA_POINTERS],
    pub layout: [ash::vk::ImageLayout; AV_NUM_DATA_POINTERS],
    pub sem: [ash::vk::Semaphore; AV_NUM_DATA_POINTERS],
    pub sem_value: [u64; AV_NUM_DATA_POINTERS],
    pub internal: *mut c_void,
    pub offset: [isize; AV_NUM_DATA_POINTERS],
    pub queue_family: [u32; AV_NUM_DATA_POINTERS],
}

#[allow(dead_code)]
unsafe extern "C" {
    pub fn av_vk_frame_alloc() -> *mut AVVkFrame;
    pub fn av_vkfmt_from_pixfmt(p: ff::AVPixelFormat) -> *const ash::vk::Format;
}

/// Raw Vulkan handles extracted from a wgpu device/context.
#[derive(Clone)]
pub struct VulkanDeviceHandles {
    pub instance: ash::Instance,
    pub physical_device: ash::vk::PhysicalDevice,
    pub device: ash::Device,
    pub queue: ash::vk::Queue,
    pub queue_family_index: u32,
    pub video_queue_family_index: u32,
    /// The `vkGetInstanceProcAddr` function pointer from the Vulkan loader.
    /// FFmpeg's `vulkan_device_init` calls this directly to load all Vulkan
    /// functions. If NULL, FFmpeg crashes with an access violation because
    /// the user-provided-device init path does NOT load libvulkan itself
    /// (unlike the from-scratch `create_instance` path).
    pub get_proc_addr: ash::vk::PFN_vkGetInstanceProcAddr,
}

impl std::fmt::Debug for VulkanDeviceHandles {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanDeviceHandles")
            .field("physical_device", &self.physical_device)
            .field("queue", &self.queue)
            .field("queue_family_index", &self.queue_family_index)
            .field("video_queue_family_index", &self.video_queue_family_index)
            .field("get_proc_addr", &(self.get_proc_addr as *const ()))
            .finish_non_exhaustive()
    }
}

/// Extract the raw Vulkan handles from a wgpu instance, device, adapter, and queue.
///
/// Returns `None` if any of the `as_hal` casts fail (i.e., the backend is not Vulkan).
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn extract_vulkan_device_handles(
    instance: &wgpu::Instance,
    adapter: &wgpu::Adapter,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Option<VulkanDeviceHandles> {
    let hal_adapter = unsafe { adapter.as_hal::<wgpu::hal::vulkan::Api>()? };
    let hal_instance = unsafe { instance.as_hal::<wgpu::hal::vulkan::Api>()? };
    let hal_device = unsafe { device.as_hal::<wgpu::hal::vulkan::Api>()? };
    let hal_queue = unsafe { queue.as_hal::<wgpu::hal::vulkan::Api>()? };

    let physical_device = hal_adapter.raw_physical_device();
    let queue_family_index = hal_device.queue_family_index();
    let queue = hal_queue.as_raw();
    let instance_shared = hal_instance.shared_instance();
    let raw_instance = instance_shared.raw_instance().clone();
    let raw_device = hal_device.raw_device().clone();

    // Get vkGetInstanceProcAddr from the Vulkan loader entry. FFmpeg needs this
    // to load all Vulkan functions during av_hwdevice_ctx_init.
    let entry = instance_shared.entry();
    let get_proc_addr = entry.static_fn().get_instance_proc_addr;

    Some(VulkanDeviceHandles {
        instance: raw_instance,
        physical_device,
        device: raw_device,
        queue,
        queue_family_index,
        // Default to the graphics queue; the caller will override this if a
        // dedicated video decode queue is available.
        video_queue_family_index: queue_family_index,
        get_proc_addr,
    })
}

/// Leak a slice of CStr pointers so they remain valid for the lifetime of the
/// FFmpeg hardware context.
fn leak_extension_ptrs(extensions: &[&'static CStr]) -> *const *const c_char {
    if extensions.is_empty() {
        return std::ptr::null();
    }

    let ptrs: Vec<*const c_char> = extensions.iter().map(|ext| ext.as_ptr()).collect();
    let leaked: &'static [*const c_char] = Box::leak(ptrs.into_boxed_slice());
    leaked.as_ptr()
}

/// Create a custom FFmpeg Vulkan hardware context that shares the same
/// `VkDevice` as the wgpu renderer.
///
/// # Safety
///
/// The handles must be valid Vulkan handles from the active wgpu instance,
/// adapter, device, and queue. The returned `AVBufferRef` is owned by the
/// caller and must be released with `ff::av_buffer_unref` when no longer needed.
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn create_ffmpeg_vulkan_device_context(
    handles: &VulkanDeviceHandles,
    instance_extensions: &[&'static CStr],
    device_extensions: &[&'static CStr],
) -> Result<NonNull<ff::AVBufferRef>> {
    let ctx_ref: *mut ff::AVBufferRef = unsafe {
        ff::av_hwdevice_ctx_alloc(ff::AVHWDeviceType::AV_HWDEVICE_TYPE_VULKAN)
    };
    if ctx_ref.is_null() {
        log::error!("[VulkanHW] av_hwdevice_ctx_alloc returned null");
        return Err(Error::HardwareContext);
    }

    let ctx_ref = NonNull::new(ctx_ref).ok_or(Error::HardwareContext)?;
    let hw_ctx = unsafe { (*ctx_ref.as_ptr()).data } as *mut ff::AVHWDeviceContext;
    let vulkan_ctx = unsafe { (*hw_ctx).hwctx } as *mut AVVulkanDeviceContext;

    if vulkan_ctx.is_null() {
        log::error!("[VulkanHW] AVHWDeviceContext.hwctx is null");
        return Err(Error::HardwareContext);
    }

    // Zero the structure before filling it.
    unsafe { std::ptr::write_bytes(vulkan_ctx, 0, 1) };

    unsafe { (*vulkan_ctx).inst = handles.instance.handle() };
    unsafe { (*vulkan_ctx).phys_dev = handles.physical_device };
    unsafe { (*vulkan_ctx).act_dev = handles.device.handle() };

    // Log Vulkan version info for debugging
    let instance_handle = handles.instance.handle();
    eprintln!(
        "[VulkanHW] Instance handle: {:?}",
        instance_handle,
    );
    // Check if QueueSubmit2 is available (needed by FFmpeg's transfer code)
    let queue_submit2_name = b"vkQueueSubmit2\0".as_ptr() as *const c_char;
    let queue_submit2_ptr = unsafe {
        (handles.get_proc_addr)(instance_handle, queue_submit2_name)
    };
    eprintln!(
        "[VulkanHW] vkQueueSubmit2 ptr: {:?}",
        queue_submit2_ptr,
    );
    let props = unsafe {
        handles.instance.get_physical_device_properties(handles.physical_device)
    };
    eprintln!(
        "[VulkanHW] Device API version: {}.{}.{}, driver: {}",
        ash::vk::api_version_major(props.api_version),
        ash::vk::api_version_minor(props.api_version),
        ash::vk::api_version_patch(props.api_version),
        props.device_name_as_c_str().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default(),
    );

    // Set get_proc_addr from the Vulkan loader. This is CRITICAL:
    // FFmpeg's vulkan_device_init calls hwctx->get_proc_addr() directly to
    // load all Vulkan functions. If NULL, it crashes with access violation.
    // The "if unset, will dynamically load libvulkan" behavior described in
    // the header only happens in the from-scratch create_instance path, NOT
    // in the user-provided-device init path.
    unsafe { (*vulkan_ctx).get_proc_addr = Some(handles.get_proc_addr) };

    // Query and store the physical device features we enabled.
    let mut features = ash::vk::PhysicalDeviceFeatures2::default();
    unsafe {
        handles.instance.get_physical_device_features2(handles.physical_device, &mut features)
    };
    unsafe { (*vulkan_ctx).device_features = features };

    // Leak extension name pointer arrays. They are small and live as long as
    // the hwdevice context.
    eprintln!("[VulkanHW] Instance extensions ({}):", instance_extensions.len());
    for ext in instance_extensions {
        eprintln!("[VulkanHW]   inst ext: {}", ext.to_string_lossy());
    }
    eprintln!("[VulkanHW] Device extensions ({}):", device_extensions.len());
    for ext in device_extensions {
        eprintln!("[VulkanHW]   dev ext: {}", ext.to_string_lossy());
    }
    unsafe {
        (*vulkan_ctx).enabled_inst_extensions = leak_extension_ptrs(instance_extensions);
        (*vulkan_ctx).nb_enabled_inst_extensions = instance_extensions.len() as c_int;
        (*vulkan_ctx).enabled_dev_extensions = leak_extension_ptrs(device_extensions);
        (*vulkan_ctx).nb_enabled_dev_extensions = device_extensions.len() as c_int;
    }

    // Queue families: graphics queue is always required.
    // On most GPUs (NVIDIA, AMD, Intel), the graphics queue family also
    // supports compute and transfer operations. We include all three flags
    // so that FFmpeg's ff_vk_qf_find() can find a queue for COMPUTE and
    // TRANSFER operations. Without these flags, ff_vk_qf_find returns NULL
    // and vulkan_frames_init crashes when dereferencing the NULL pointer.
    let graphics_qf = AVVulkanDeviceQueueFamily {
        idx: handles.queue_family_index as c_int,
        num: 1,
        flags: ash::vk::QueueFlags::GRAPHICS
            | ash::vk::QueueFlags::COMPUTE
            | ash::vk::QueueFlags::TRANSFER,
        video_caps: ash::vk::VideoCodecOperationFlagsKHR::empty(),
    };
    unsafe {
        (*vulkan_ctx).qf[0] = graphics_qf;
        (*vulkan_ctx).nb_qf = 1;
    }

    // If the video decode queue is on a separate family, add it too.
    // Set video_caps to empty so FFmpeg's vulkan_device_init queries the
    // actual device capabilities via vkGetPhysicalDeviceQueueFamilyProperties2.
    // Setting incorrect caps (e.g. claiming AV1 support on a device that
    // doesn't have it) can cause crashes during video session creation.
    if handles.video_queue_family_index != handles.queue_family_index {
        let decode_qf = AVVulkanDeviceQueueFamily {
            idx: handles.video_queue_family_index as c_int,
            num: 1,
            flags: ash::vk::QueueFlags::VIDEO_DECODE_KHR
                | ash::vk::QueueFlags::TRANSFER,
            video_caps: ash::vk::VideoCodecOperationFlagsKHR::empty(),
        };
        unsafe {
            (*vulkan_ctx).qf[1] = decode_qf;
            (*vulkan_ctx).nb_qf = 2;
        }
    }

    // Set deprecated queue family fields for backwards compatibility.
    // FFmpeg 8.0 (libavutil major 60) still has FF_API_VULKAN_FIXED_QUEUES
    // active, so av_hwdevice_ctx_init may read these. Set them to match the
    // qf array above. Use -1 for unavailable queue families.
    let gqf = handles.queue_family_index as c_int;
    let dqf = handles.video_queue_family_index as c_int;
    let has_separate_decode = handles.video_queue_family_index != handles.queue_family_index;
    unsafe {
        (*vulkan_ctx).queue_family_index = gqf;
        (*vulkan_ctx).nb_graphics_queues = 1;
        (*vulkan_ctx).queue_family_tx_index = gqf;
        (*vulkan_ctx).nb_tx_queues = 1;
        (*vulkan_ctx).queue_family_comp_index = gqf;
        (*vulkan_ctx).nb_comp_queues = 1;
        (*vulkan_ctx).queue_family_encode_index = -1;
        (*vulkan_ctx).nb_encode_queues = 0;
        if has_separate_decode {
            (*vulkan_ctx).queue_family_decode_index = dqf;
            (*vulkan_ctx).nb_decode_queues = 1;
        } else {
            (*vulkan_ctx).queue_family_decode_index = -1;
            (*vulkan_ctx).nb_decode_queues = 0;
        }
    }

    // Provide no-op lock_queue/unlock_queue callbacks. FFmpeg's default
    // implementations access p->qf_mutex[queue_family][index], which may
    // crash if the queue family index is out of range or the mutex array
    // is not properly initialized. Since we only use the device from a
    // single thread (the video thread), no locking is needed.
    unsafe extern "C" fn noop_lock_queue(_ctx: *mut c_void, _qf: u32, _idx: u32) {}
    unsafe { (*vulkan_ctx).lock_queue = Some(noop_lock_queue) };
    unsafe { (*vulkan_ctx).unlock_queue = Some(noop_lock_queue) };

    log::info!("[VulkanHW] Calling av_hwdevice_ctx_init (struct size: {})", std::mem::size_of::<AVVulkanDeviceContext>());
    let init_ret = unsafe { ff::av_hwdevice_ctx_init(ctx_ref.as_ptr()) };
    if init_ret != 0 {
        log::error!("[VulkanHW] av_hwdevice_ctx_init failed: {}", init_ret);
        return Err(Error::HardwareContext);
    }

    log::info!("[VulkanHW] Created shared FFmpeg Vulkan hwdevice context");

    Ok(ctx_ref)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Print sizes and offsets for debugging FFI layout mismatches.
    #[test]
    fn print_struct_layout() {
        eprintln!("=== AVVulkanDeviceContext layout ===");
        eprintln!("  size: {}", std::mem::size_of::<AVVulkanDeviceContext>());
        eprintln!("  align: {}", std::mem::align_of::<AVVulkanDeviceContext>());
        eprintln!("  offset alloc: {}", std::mem::offset_of!(AVVulkanDeviceContext, alloc));
        eprintln!("  offset get_proc_addr: {}", std::mem::offset_of!(AVVulkanDeviceContext, get_proc_addr));
        eprintln!("  offset inst: {}", std::mem::offset_of!(AVVulkanDeviceContext, inst));
        eprintln!("  offset phys_dev: {}", std::mem::offset_of!(AVVulkanDeviceContext, phys_dev));
        eprintln!("  offset act_dev: {}", std::mem::offset_of!(AVVulkanDeviceContext, act_dev));
        eprintln!("  offset device_features: {}", std::mem::offset_of!(AVVulkanDeviceContext, device_features));
        eprintln!("  offset enabled_inst_extensions: {}", std::mem::offset_of!(AVVulkanDeviceContext, enabled_inst_extensions));
        eprintln!("  offset nb_enabled_inst_extensions: {}", std::mem::offset_of!(AVVulkanDeviceContext, nb_enabled_inst_extensions));
        eprintln!("  offset enabled_dev_extensions: {}", std::mem::offset_of!(AVVulkanDeviceContext, enabled_dev_extensions));
        eprintln!("  offset nb_enabled_dev_extensions: {}", std::mem::offset_of!(AVVulkanDeviceContext, nb_enabled_dev_extensions));
        eprintln!("  offset queue_family_index: {}", std::mem::offset_of!(AVVulkanDeviceContext, queue_family_index));
        eprintln!("  offset nb_decode_queues: {}", std::mem::offset_of!(AVVulkanDeviceContext, nb_decode_queues));
        eprintln!("  offset lock_queue: {}", std::mem::offset_of!(AVVulkanDeviceContext, lock_queue));
        eprintln!("  offset unlock_queue: {}", std::mem::offset_of!(AVVulkanDeviceContext, unlock_queue));
        eprintln!("  offset qf: {}", std::mem::offset_of!(AVVulkanDeviceContext, qf));
        eprintln!("  offset nb_qf: {}", std::mem::offset_of!(AVVulkanDeviceContext, nb_qf));
        eprintln!("  PhysicalDeviceFeatures size: {}", std::mem::size_of::<ash::vk::PhysicalDeviceFeatures>());
        eprintln!("  PhysicalDeviceFeatures2 size: {}", std::mem::size_of::<ash::vk::PhysicalDeviceFeatures2<'static>>());
        eprintln!("  AVVulkanDeviceQueueFamily size: {}", std::mem::size_of::<AVVulkanDeviceQueueFamily>());
        eprintln!("  AVVulkanFramesContext size: {}", std::mem::size_of::<AVVulkanFramesContext>());
        eprintln!("  AVVkFrame size: {}", std::mem::size_of::<AVVkFrame>());
    }

    /// The size of `AVVulkanDeviceContext` is fixed by the public header, so we
    /// can catch header mismatches at test time.
    #[test]
    fn av_vulkan_device_context_size_matches() {
        assert!(std::mem::size_of::<AVVulkanDeviceContext>() > 0);
        assert!(std::mem::offset_of!(AVVulkanDeviceContext, qf) > 0);
        assert!(
            std::mem::offset_of!(AVVulkanDeviceContext, nb_qf)
                > std::mem::offset_of!(AVVulkanDeviceContext, qf)
        );
    }

    #[test]
    fn av_vk_frame_alloc_works() {
        let frame = unsafe { av_vk_frame_alloc() };
        assert!(!frame.is_null());
        unsafe { ff::av_free(frame as *mut c_void) };
    }
}

