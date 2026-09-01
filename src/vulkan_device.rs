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
    c"VK_KHR_video_maintenance1",
    c"VK_KHR_video_maintenance2",
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
    c"VK_KHR_external_memory_fd",
    c"VK_KHR_external_semaphore_fd",
    c"VK_EXT_external_memory_host",
    c"VK_EXT_external_memory_dma_buf",
    c"VK_EXT_image_drm_format_modifier",
    c"VK_EXT_physical_device_drm",
];

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

    let supported_names: std::collections::HashSet<String> = supported
        .iter()
        .filter_map(|p| {
            p.extension_name_as_c_str()
                .ok()
                .map(|s| s.to_string_lossy().into_owned())
        })
        .collect();

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

pub struct VulkanVideoDevice {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub video_queue_family_index: u32,
}

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
    extensions.sort_unstable_by(|a, b| a.to_bytes().cmp(b.to_bytes()));
    extensions.dedup_by(|a, b| a.to_bytes() == b.to_bytes());
    extensions
}

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
pub fn create_vulkan_device_for_video(
    instance: &wgpu::Instance,
    adapter: &wgpu::Adapter,
    required_features: wgpu::Features,
    required_limits: &wgpu::Limits,
) -> Result<VulkanVideoDevice> {
    let hal_adapter = unsafe {
        adapter
            .as_hal::<wgpu::hal::vulkan::Api>()
            .ok_or(Error::UnsupportedBackend)?
    };

    let instance_hal = unsafe {
        instance
            .as_hal::<wgpu::hal::vulkan::Api>()
            .ok_or(Error::UnsupportedBackend)?
    };

    let instance_shared = instance_hal.shared_instance();
    let raw_instance = instance_shared.raw_instance();
    let physical_device = hal_adapter.raw_physical_device();

    let queue_families =
        unsafe { raw_instance.get_physical_device_queue_family_properties(physical_device) };

    // Prefer a video-decode family that does not also carry graphics work.
    // Keeping decode off the renderer queue avoids unnecessary cross-library
    // VkQueue contention between FFmpeg and wgpu. If the device exposes only
    // a combined family, retain it as the compatibility fallback.
    let dedicated_video_qfi = queue_families
        .iter()
        .enumerate()
        .find(|(_, props)| {
            props
                .queue_flags
                .contains(ash::vk::QueueFlags::VIDEO_DECODE_KHR)
                && !props.queue_flags.contains(ash::vk::QueueFlags::GRAPHICS)
        })
        .map(|(i, _)| i as u32);

    let video_queue_family_index = dedicated_video_qfi
        .or_else(|| {
            queue_families
                .iter()
                .enumerate()
                .find(|(_, props)| {
                    props
                        .queue_flags
                        .contains(ash::vk::QueueFlags::VIDEO_DECODE_KHR)
                })
                .map(|(i, _)| i as u32)
        })
        .ok_or_else(|| {
            log::error!("[VulkanVideo] No queue family with VIDEO_DECODE_KHR found");
            Error::HardwareContext
        })?;

    log::info!(
        "[VulkanVideo] Selected video decode queue family: {}{}",
        video_queue_family_index,
        if dedicated_video_qfi.is_some() {
            " (dedicated)"
        } else {
            " (combined fallback)"
        }
    );

    let supported_video_exts = supported_video_decode_extensions(raw_instance, physical_device);
    let video_qfi = video_queue_family_index;
    const QUEUE_PRIORITY: &[f32] = &[1.0];

    let callback = Box::new(move |args: wgpu::hal::vulkan::CreateDeviceCallbackArgs| {
        for ext in &supported_video_exts {
            if !args
                .extensions
                .iter()
                .any(|enabled| enabled.to_bytes() == ext.to_bytes())
            {
                log::debug!("[VulkanVideo] Enabling extension: {:?}", ext);
                args.extensions.push(*ext);
            }
        }

        // Use wgpu-hal's actual VkDevice queue-create infos as the authority.
        // Do not independently guess which graphics family wgpu selected.
        let video_family_already_requested = args
            .queue_create_infos
            .iter()
            .any(|info| info.queue_family_index == video_qfi);

        if !video_family_already_requested {
            let video_queue_info = ash::vk::DeviceQueueCreateInfo::default()
                .queue_family_index(video_qfi)
                .queue_priorities(QUEUE_PRIORITY);
            args.queue_create_infos.push(video_queue_info);
        }
    });

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
