pub mod context;
pub(crate) mod decode;
pub(crate) mod error;
pub(crate) mod probe;
#[cfg(feature = "video")]
pub(crate) mod software_video;
pub(crate) mod video;
pub mod video_caps;
pub mod cuda_interop;
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
    SoftwareFrameReceiver, YuvRange,
};
pub use video::{SeekMode, Video};
pub use vulkan_device::{VulkanVideoDevice, create_vulkan_device_for_video};

pub fn required_wgpu_device_features(adapter: &wgpu::Adapter) -> wgpu::Features {
    let mut desired = match adapter.get_info().backend {
        wgpu::Backend::Vulkan | wgpu::Backend::Dx12 => {
            wgpu::Features::TEXTURE_FORMAT_NV12
                | wgpu::Features::TEXTURE_FORMAT_P010
                | wgpu::Features::TEXTURE_FORMAT_16BIT_NORM
        }
        _ => wgpu::Features::empty(),
    };
    #[cfg(target_os = "windows")]
    if adapter.get_info().backend == wgpu::Backend::Vulkan {
        desired |= wgpu::Features::VULKAN_EXTERNAL_MEMORY_WIN32;
    }
    desired & adapter.features()
}
