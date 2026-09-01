pub mod context;
pub(crate) mod decode;
pub(crate) mod error;
pub(crate) mod probe;
#[cfg(feature = "video")]
pub(crate) mod software_video;
pub(crate) mod video;
pub(crate) mod vulkan_device;

pub use context::Context;
pub use context::layout;
pub use decode::{
    audio::{AudioMetadata, AudioParameters, AudioSink, DeviceAudioSink},
    video::{DiscardLevel, VideoMetadata},
    vulkan_hwcontext::VulkanDeviceHandles,
};
pub use error::{Error, Result};
pub use probe::{probe_hardware_decoding_support, probe_software_required};
#[cfg(feature = "video")]
pub use software_video::{
    ColorMatrix, MasterClock, PixelFormat, SoftwareContext, SoftwareDecodeVideo, SoftwareFrame,
    SoftwareFrameReceiver, SoftwareVideo, YuvRange,
};
pub use video::{SeekMode, Video};
pub use vulkan_device::{VulkanVideoDevice, create_vulkan_device_for_video};

#[cfg(target_os = "windows")]
pub fn required_wgpu_device_features(adapter: &wgpu::Adapter) -> wgpu::Features {
    match adapter.get_info().backend {
        wgpu::Backend::Vulkan => {
            wgpu::Features::TEXTURE_FORMAT_NV12
                | wgpu::Features::TEXTURE_FORMAT_P010
                | wgpu::Features::TEXTURE_FORMAT_16BIT_NORM
                | wgpu::Features::VULKAN_EXTERNAL_MEMORY_WIN32
        }
        wgpu::Backend::Dx12 => {
            wgpu::Features::TEXTURE_FORMAT_NV12
                | wgpu::Features::TEXTURE_FORMAT_P010
                | wgpu::Features::TEXTURE_FORMAT_16BIT_NORM
        }
        _ => wgpu::Features::empty(),
    }
}

#[cfg(target_os = "linux")]
pub fn required_wgpu_device_features(adapter: &wgpu::Adapter) -> wgpu::Features {
    match adapter.get_info().backend {
        wgpu::Backend::Vulkan => {
            wgpu::Features::TEXTURE_FORMAT_NV12
                | wgpu::Features::TEXTURE_FORMAT_P010
                | wgpu::Features::TEXTURE_FORMAT_16BIT_NORM
        }
        _ => wgpu::Features::empty(),
    }
}

#[cfg(target_os = "macos")]
pub fn required_wgpu_device_features(_adapter: &wgpu::Adapter) -> wgpu::Features {
    wgpu::Features::empty()
}
