//! Vulkan device creation with video decode extensions
//!
//! This module provides a function to create a wgpu Device with Vulkan Video
//! decode extensions enabled, allowing FFmpeg to use the same VkDevice for
//! hardware-accelerated video decoding.

use crate::error::{Error, Result};

/// Vulkan Video decode extensions to enable on the VkDevice.
///
/// These are added to the extension list inside the `open_with_callback`
/// callback, which is the only way to extend the extensions list that
/// `vkCreateDevice` actually sees (the `create_info` field is overwritten
/// by wgpu-hal after the callback returns).
///
/// Note: VK_KHR_video_decode_vp9 is not in ash 0.38 (added in Vulkan 1.4.317).
/// If VP9 decode support is needed, the extension name can be passed as a raw
/// CStr: `c"VK_KHR_video_decode_vp9"`.
const VIDEO_DECODE_EXTENSIONS: &[&std::ffi::CStr] = &[
    ash::vk::KHR_VIDEO_QUEUE_NAME,
    ash::vk::KHR_VIDEO_DECODE_QUEUE_NAME,
    ash::vk::KHR_VIDEO_DECODE_H264_NAME,
    ash::vk::KHR_VIDEO_DECODE_H265_NAME,
    ash::vk::KHR_VIDEO_DECODE_AV1_NAME,
];

/// Optional device extensions that FFmpeg's `vulkan_device_create` would
/// normally enable if supported. These are needed for FFmpeg's Vulkan backend
/// to function correctly — without them, FFmpeg may have NULL function pointers
/// for extensions it expects to be available, causing access violations during
/// decoding.
///
/// This list mirrors the `optional_device_exts` array in FFmpeg's
/// `libavutil/hwcontext_vulkan.c`.
const OPTIONAL_FFMPEG_EXTENSIONS: &[&std::ffi::CStr] = &[
    // Video maintenance extensions — needed for video decode
    c"VK_KHR_video_maintenance1",
    c"VK_KHR_video_maintenance2",
    // General extensions FFmpeg uses for its Vulkan backend
    c"VK_KHR_portability_subset",
    c"VK_KHR_push_descriptor",
    c"VK_EXT_descriptor_buffer",
    c"VK_EXT_shader_atomic_float",
    c"VK_KHR_cooperative_matrix",
    c"VK_EXT_shader_object",
    c"VK_KHR_shader_subgroup_rotate",
    c"VK_KHR_shader_expect_assume",
    c"VK_EXT_host_image_copy",
    c"VK_KHR_shader_relaxed_extended_instruction",
    // External memory/semaphore extensions for interop
    c"VK_KHR_external_memory_fd",
    c"VK_KHR_external_semaphore_fd",
    c"VK_EXT_external_memory_host",
    c"VK_EXT_external_memory_dma_buf",
    c"VK_EXT_image_drm_format_modifier",
    c"VK_EXT_physical_device_drm",
];

/// Query the physical device for its supported extensions and return the
/// subset of `VIDEO_DECODE_EXTENSIONS` + `OPTIONAL_FFMPEG_EXTENSIONS` that are
/// actually available.
///
/// This is critical: requesting an extension the device doesn't support causes
/// `vkCreateDevice` to return `ERROR_EXTENSION_NOT_PRESENT`, which wgpu-hal
/// turns into a panic (not a `Result::Err`).
fn supported_video_decode_extensions(
    raw_instance: &ash::Instance,
    physical_device: ash::vk::PhysicalDevice,
) -> Vec<&'static std::ffi::CStr> {
    let supported =
        match unsafe { raw_instance.enumerate_device_extension_properties(physical_device) } {
            Ok(props) => props,
            Err(e) => {
                log::warn!(
                    "[VulkanVideo] Failed to enumerate device extensions: {:?}",
                    e
                );
                return Vec::new();
            }
        };

    // Build a set of supported extension name strings for fast lookup.
    let supported_names: std::collections::HashSet<String> = supported
        .iter()
        .filter_map(|p| {
            p.extension_name_as_c_str()
                .ok()
                .map(|s| s.to_string_lossy().into_owned())
        })
        .collect();

    // Combine video decode extensions with optional FFmpeg extensions.
    let all_requested: Vec<&'static std::ffi::CStr> = VIDEO_DECODE_EXTENSIONS
        .iter()
        .chain(OPTIONAL_FFMPEG_EXTENSIONS.iter())
        .copied()
        .collect();

    let result: Vec<&'static std::ffi::CStr> = all_requested
        .iter()
        .filter(|ext| supported_names.contains(ext.to_string_lossy().as_ref()))
        .copied()
        .collect();

    let skipped: Vec<&std::ffi::CStr> = all_requested
        .iter()
        .filter(|ext| !supported_names.contains(ext.to_string_lossy().as_ref()))
        .copied()
        .collect();

    eprintln!(
        "[VulkanVideo] Supported extensions for FFmpeg: {:?}",
        result
            .iter()
            .map(|e| e.to_string_lossy())
            .collect::<Vec<_>>()
    );
    if !skipped.is_empty() {
        eprintln!(
            "[VulkanVideo] Skipped unsupported extensions: {:?}",
            skipped
                .iter()
                .map(|e| e.to_string_lossy())
                .collect::<Vec<_>>()
        );
    }

    result
}

/// Result of creating a Vulkan video device.
pub struct VulkanVideoDevice {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub video_queue_family_index: u32,
}

/// Return the full list of device extensions that should be enabled when
/// creating a shared Vulkan device for video decoding. This includes the
/// extensions wgpu requires plus the Vulkan Video decode extensions that are
/// actually supported by the physical device.
pub fn enabled_device_extensions(adapter: &wgpu::Adapter) -> Vec<&'static std::ffi::CStr> {
    let hal_adapter = unsafe { adapter.as_hal::<wgpu::hal::vulkan::Api>() };

    let mut extensions: Vec<&'static std::ffi::CStr> = Vec::new();
    let mut video_exts: Vec<&'static std::ffi::CStr> = Vec::new();

    if let Some(ha) = hal_adapter {
        extensions = ha.required_device_extensions(wgpu::Features::empty());
        let instance_shared = ha.shared_instance();
        let raw_instance = instance_shared.raw_instance();
        let physical_device = ha.raw_physical_device();
        video_exts = supported_video_decode_extensions(raw_instance, physical_device);
    }

    extensions.extend(video_exts);
    extensions
}

/// Return the list of instance extensions enabled on the wgpu Vulkan instance.
pub fn enabled_instance_extensions(instance: &wgpu::Instance) -> Vec<&'static std::ffi::CStr> {
    let hal_instance = unsafe { instance.as_hal::<wgpu::hal::vulkan::Api>() };
    hal_instance
        .map(|i| {
            let shared = i.shared_instance();
            shared.extensions().to_vec()
        })
        .unwrap_or_default()
}

/// Create a wgpu Device with Vulkan Video decode extensions enabled.
///
/// This function:
/// 1. Gets the hal adapter for Vulkan
/// 2. Queries queue families for video decode support
/// 3. Uses wgpu-hal's callback mechanism to:
///    a. Add video decode extensions to the enabled extension list
///    b. Add a video decode queue if it's on a different queue family
/// 4. Creates the device with the custom configuration
/// 5. Wraps it as a wgpu Device
///
/// # Returns
/// - `Ok(VulkanVideoDevice)` if successful
/// - `Err(Error::UnsupportedBackend)` if not on Vulkan backend
/// - `Err(Error::HardwareContext)` if video decode queue not found or device creation fails
pub fn create_vulkan_device_for_video(
    instance: &wgpu::Instance,
    adapter: &wgpu::Adapter,
    required_features: wgpu::Features,
    required_limits: &wgpu::Limits,
) -> Result<VulkanVideoDevice> {
    // Get the hal adapter for Vulkan
    let hal_adapter = unsafe {
        adapter
            .as_hal::<wgpu::hal::vulkan::Api>()
            .ok_or(Error::UnsupportedBackend)?
    };

    // Get instance and physical device for queue family query
    let instance_hal = unsafe {
        instance
            .as_hal::<wgpu::hal::vulkan::Api>()
            .ok_or(Error::UnsupportedBackend)?
    };

    let instance_shared = instance_hal.shared_instance();
    let raw_instance = instance_shared.raw_instance();
    let physical_device = hal_adapter.raw_physical_device();

    // Query queue families to find one with video decode support
    let queue_families =
        unsafe { raw_instance.get_physical_device_queue_family_properties(physical_device) };

    let mut video_queue_family_index = None;
    let mut graphics_queue_family_index = None;

    // First pass: prefer a queue family that has both graphics and video decode
    for (i, props) in queue_families.iter().enumerate() {
        let i = i as u32;
        if props.queue_flags.contains(ash::vk::QueueFlags::GRAPHICS)
            && props
                .queue_flags
                .contains(ash::vk::QueueFlags::VIDEO_DECODE_KHR)
        {
            video_queue_family_index = Some(i);
            graphics_queue_family_index = Some(i);
            break;
        }
        if props.queue_flags.contains(ash::vk::QueueFlags::GRAPHICS) {
            graphics_queue_family_index = Some(i);
        }
    }

    // If no combined queue, look for a separate video decode queue
    if video_queue_family_index.is_none() {
        for (i, props) in queue_families.iter().enumerate() {
            let i = i as u32;
            if props
                .queue_flags
                .contains(ash::vk::QueueFlags::VIDEO_DECODE_KHR)
            {
                video_queue_family_index = Some(i);
                break;
            }
        }
    }

    let video_queue_family_index = video_queue_family_index.ok_or_else(|| {
        log::error!("[VulkanVideo] No queue family with VIDEO_DECODE_KHR found");
        Error::HardwareContext
    })?;

    let graphics_queue_family_index = graphics_queue_family_index.ok_or_else(|| {
        log::error!("[VulkanVideo] No queue family with GRAPHICS found");
        Error::HardwareContext
    })?;

    log::info!(
        "[VulkanVideo] Graphics queue family: {}, Video decode queue family: {}",
        graphics_queue_family_index,
        video_queue_family_index
    );

    // Use wgpu-hal's callback mechanism to customize device creation.
    //
    // The callback receives `args.extensions` which is the Vec<&'static CStr>
    // that wgpu-hal will actually pass to vkCreateDevice. We must add our
    // video decode extensions THERE, not to a separate Vec.
    //
    // Similarly, `args.queue_create_infos` is the Vec that gets passed to
    // vkCreateDevice, so we add the video decode queue there if it's on a
    // different queue family than the graphics queue.
    // Query which video decode extensions are actually supported by this
    // physical device. Requesting unsupported extensions causes vkCreateDevice
    // to return ERROR_EXTENSION_NOT_PRESENT, which wgpu-hal turns into a panic.
    let supported_video_exts = supported_video_decode_extensions(raw_instance, physical_device);

    let video_qfi = video_queue_family_index;
    let graphics_qfi = graphics_queue_family_index;

    // Static queue priority - must be 'static to avoid dangling reference
    // after the callback returns (the DeviceQueueCreateInfo in the Vec is used
    // by open_with_callback after the callback completes).
    const QUEUE_PRIORITY: &[f32] = &[1.0];

    let callback = Box::new(move |args: wgpu::hal::vulkan::CreateDeviceCallbackArgs| {
        // Add only the video decode extensions that the physical device supports.
        for ext in &supported_video_exts {
            log::debug!("[VulkanVideo] Enabling extension: {:?}", ext);
            args.extensions.push(*ext);
        }

        // Add video decode queue if it's on a different queue family than
        // the graphics queue that wgpu-hal already added.
        if video_qfi != graphics_qfi {
            let video_queue_info = ash::vk::DeviceQueueCreateInfo::default()
                .queue_family_index(video_qfi)
                .queue_priorities(QUEUE_PRIORITY);
            args.queue_create_infos.push(video_queue_info);
        }
    });

    // Open the device with our callback
    let hal_device = unsafe {
        hal_adapter
            .open_with_callback(
                required_features,
                required_limits,
                &wgpu::MemoryHints::default(),
                Some(callback),
            )
            .map_err(|e| {
                log::error!("[VulkanVideo] Failed to open device: {:?}", e);
                Error::HardwareContext
            })?
    };

    // Wrap the hal device as a wgpu device
    let (device, queue) = unsafe {
        adapter
            .create_device_from_hal(
                hal_device,
                &wgpu::DeviceDescriptor {
                    label: Some("NexaEngine Vulkan Video Device"),
                    required_features,
                    required_limits: required_limits.clone(),
                    memory_hints: wgpu::MemoryHints::default(),
                    experimental_features: wgpu::ExperimentalFeatures::default(),
                    trace: wgpu::Trace::Off,
                },
            )
            .map_err(|e| {
                log::error!("[VulkanVideo] Failed to create wgpu device: {:?}", e);
                Error::HardwareContext
            })?
    };

    log::info!("[VulkanVideo] Successfully created device with video decode support");

    Ok(VulkanVideoDevice {
        device,
        queue,
        video_queue_family_index,
    })
}
