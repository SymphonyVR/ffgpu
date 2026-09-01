pub(crate) mod layout;
pub(crate) mod pipeline_cache;

use crate::{
    decode::{audio::AudioSink, vulkan_hwcontext},
    error::Result,
    video::Video,
    vulkan_device,
};
use pipeline_cache::PipelineCache;
use std::{
    path::Path,
    ptr::NonNull,
    sync::{Arc, Mutex},
};

/// Send+Sync wrapper around an FFmpeg `AVBufferRef` pointer.
///
/// The pointer is only accessed from the thread that owns the `Context`, and
/// FFmpeg reference-counted buffers are thread-safe to reference, so we can
/// mark the pointer as Send+Sync.
#[derive(Clone, Copy)]
struct AvBufferRefPtr(*mut ffmpeg_next::sys::AVBufferRef);

unsafe impl Send for AvBufferRefPtr {}
unsafe impl Sync for AvBufferRefPtr {}

pub struct Context {
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline_cache: Arc<Mutex<PipelineCache>>,
    /// Shared FFmpeg Vulkan hardware device context, created once and reused for
    /// every video opened through this context. This avoids leaking extension
    /// name arrays and re-initializing the Vulkan device for each video.
    vulkan_hw_device_ctx: Option<AvBufferRefPtr>,
}

impl Context {
    pub fn new(
        instance: &wgpu::Instance,
        adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Self> {
        Self::new_with_vulkan_queue_family(instance, adapter, device, queue, None)
    }

    pub fn new_with_vulkan_queue_family(
        instance: &wgpu::Instance,
        adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        vulkan_video_queue_family_index: Option<u32>,
    ) -> Result<Self> {
        ffmpeg_next::init()?;

        let instance = instance.clone();
        let adapter = adapter.clone();
        let device = device.clone();
        let queue = queue.clone();

        let pipeline_cache = Arc::new(Mutex::new(PipelineCache::new(device.clone())));

        // Create a shared FFmpeg Vulkan hwdevice context once, so every video
        // opened through this context uses the same VkDevice as wgpu.
        let vulkan_hw_device_ctx =
            if adapter.get_info().backend == wgpu::Backend::Vulkan {
                let mut handles = unsafe {
                    vulkan_hwcontext::extract_vulkan_device_handles(
                        &instance, &adapter, &device, &queue,
                    )
                }
                .ok_or(crate::error::Error::HardwareContext)?;

                if let Some(qfi) = vulkan_video_queue_family_index {
                    handles.video_queue_family_index = qfi;
                }

                let device_extensions = vulkan_device::enabled_device_extensions(&adapter);
                let instance_extensions = vulkan_device::enabled_instance_extensions(&instance);

                let ctx = unsafe {
                    vulkan_hwcontext::create_ffmpeg_vulkan_device_context(
                        &handles,
                        &instance_extensions,
                        &device_extensions,
                    )?
                };
                Some(AvBufferRefPtr(ctx.as_ptr()))
            } else {
                None
            };

        Ok(Context {
            instance,
            adapter,
            device,
            queue,
            pipeline_cache,
            vulkan_hw_device_ctx,
        })
    }

    pub fn create_video<P>(&mut self, path: &P) -> Result<(Video, AudioSink)>
    where
        P: AsRef<Path> + ?Sized,
    {
        Video::new(
            self.instance.clone(),
            self.adapter.clone(),
            self.device.clone(),
            self.queue.clone(),
            self.pipeline_cache.clone(),
            self.vulkan_hw_device_ctx.map(|p| NonNull::new(p.0).unwrap()),
            path,
        )
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        if let Some(mut ctx) = self.vulkan_hw_device_ctx {
            unsafe {
                ffmpeg_next::sys::av_buffer_unref(&mut ctx.0);
            }
        }
    }
}
