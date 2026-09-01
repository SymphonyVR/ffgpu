//! Vulkan Video frame adapter for zero-copy video decoding.
//!
//! This module imports FFmpeg's Vulkan hardware frames (AVVkFrame) into wgpu
//! textures by wrapping the raw VkImage handles. It requires that FFmpeg and
//! wgpu share the same VkDevice, which is set up in
//! `decode::vulkan_hwcontext` and `video::Video::new`.

use super::FrameAdapter;
use crate::{
    context::{layout, pipeline_cache::PipelineCache},
    decode::{
        frames::FrameAdapterBuilder,
        vulkan_hwcontext::{AVVkFrame, AVVulkanFramesContext, AV_NUM_DATA_POINTERS},
    },
    error::{Error, Result},
};
use ffmpeg_next::sys as ff;
use std::ptr::NonNull;

/// Number of frames to keep in the texture cache.
///
/// FFmpeg reuses `AVVkFrame` objects from a pool, so the same `VkImage` can
/// appear for multiple consecutive frames. We cache the imported wgpu texture
/// keyed by the `VkImage` handle and the frame dimensions, so we don't recreate
/// it every frame.
const TEXTURE_CACHE_SIZE: usize = 8;

/// A cached imported texture and its views.
struct CachedTexture {
    vk_image: ash::vk::Image,
    width: u32,
    height: u32,
    texture: wgpu::Texture,
    #[allow(dead_code)]
    identity: layout::FrameDescriptor<()>,
}

pub struct VulkanVideoFrameAdapter {
    /// Cached textures, keyed by the raw VkImage handle.
    texture_cache: std::collections::VecDeque<CachedTexture>,
    /// Currently active bind group for the last imported frame.
    bg0: Option<wgpu::BindGroup>,
    /// Identity layout of the currently active frame.
    identity: Option<layout::FrameDescriptor<()>>,
}

impl FrameAdapterBuilder for VulkanVideoFrameAdapter {
    unsafe fn new(_decoder: NonNull<ff::AVCodecContext>) -> Result<Self> {
        Ok(VulkanVideoFrameAdapter {
            texture_cache: std::collections::VecDeque::with_capacity(TEXTURE_CACHE_SIZE),
            bg0: None,
            identity: None,
        })
    }

    fn supports_format(format: ff::AVPixelFormat) -> bool {
        format == ff::AVPixelFormat::AV_PIX_FMT_VULKAN
    }
}

impl FrameAdapter for VulkanVideoFrameAdapter {
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn import_frame(
        &mut self,
        frame: NonNull<ff::AVFrame>,
        _instance: &wgpu::Instance,
        _adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _encoder: &mut wgpu::CommandEncoder,
        pipeline_cache: &mut PipelineCache,
    ) -> Result<()> {
        let frame = unsafe { frame.as_ref() };

        if frame.format != ff::AVPixelFormat::AV_PIX_FMT_VULKAN as i32 {
            return Err(Error::UnsupportedBackend);
        }

        if frame.hw_frames_ctx.is_null() {
            log::error!("[VulkanVideo] AVFrame has no hw_frames_ctx");
            return Err(Error::UnsupportedBackend);
        }

        // The AVVkFrame is stored in frame.data[0]. This is the FFmpeg convention
        // for Vulkan hardware frames.
        let vk_frame = frame.data[0] as *mut AVVkFrame;
        if vk_frame.is_null() {
            log::error!("[VulkanVideo] AVFrame.data[0] is null");
            return Err(Error::UnsupportedBackend);
        }
        let vk_frame = unsafe { &*vk_frame };

        let frames_ctx = unsafe { &*(frame.hw_frames_ctx as *mut ff::AVHWFramesContext) };
        let vulkan_frames_ctx = frames_ctx.hwctx as *mut AVVulkanFramesContext;
        if vulkan_frames_ctx.is_null() {
            log::error!("[VulkanVideo] AVHWFramesContext.hwctx is null");
            return Err(Error::UnsupportedBackend);
        }
        let vulkan_frames_ctx = unsafe { &*vulkan_frames_ctx };

        let sw_format = frames_ctx.sw_format;
        let width = frame.width as u32;
        let height = frame.height as u32;

        // Determine how many planes are valid. FFmpeg stores one VkImage per
        // plane, or a single multi-plane image. We stop at the first
        // VK_NULL_HANDLE.
        let mut plane_count = 0;
        while plane_count < AV_NUM_DATA_POINTERS
            && vk_frame.img[plane_count] != ash::vk::Image::null()
        {
            plane_count += 1;
        }

        if plane_count == 0 {
            log::error!("[VulkanVideo] AVVkFrame has no images");
            return Err(Error::UnsupportedBackend);
        }

        // Determine the wgpu texture format for each plane. For multi-plane
        // Vulkan images, the software format determines the plane layout.
        let texture_format = layout::vk_format_texture_format(
            vulkan_frames_ctx.format[0],
            ff::AVPixelFormat::try_from(sw_format).unwrap_or(ff::AVPixelFormat::AV_PIX_FMT_NONE),
        )
        .ok_or_else(|| {
            log::error!(
                "[VulkanVideo] Unsupported Vulkan format {:?} / sw_format {:?}",
                vulkan_frames_ctx.format[0],
                sw_format
            );
            Error::UnsupportedBackend
        })?;

        // For the first iteration, we synchronize by idling the queue. This
        // guarantees that the decode work (and any previous sampling work) is
        // complete before we sample the imported image. In a later iteration,
        // this should be replaced with a proper timeline semaphore wait.
        device.poll(wgpu::PollType::wait_indefinitely()).ok();

        // Import each VkImage as a wgpu::Texture. We cache by VkImage handle so
        // that repeated frames from the same pool do not recreate textures.
        let mut textures = Vec::with_capacity(plane_count);
        for i in 0..plane_count {
            let vk_image = vk_frame.img[i];

            if let Some(cached) = self.texture_cache.iter().find(|c| {
                c.vk_image == vk_image && c.width == width && c.height == height
            }) {
                textures.push(cached.texture.clone());
                continue;
            }

            let hal_device = device.as_hal::<wgpu::hal::vulkan::Api>().ok_or(Error::UnsupportedBackend)?;

            let plane_format = match &texture_format.planes {
                layout::PlaneLayout::PackedYUV420([y, _uv]) if i == 0 => *y,
                layout::PlaneLayout::PackedYUV420([_y, uv]) if i == 1 => *uv,
                layout::PlaneLayout::PackedYUV420([y, _uv]) => {
                    // Fallback for any additional planes
                    *y
                }
                layout::PlaneLayout::YUV420([y, u, v]) => match i {
                    0 => *y,
                    1 => *u,
                    2 => *v,
                    _ => *y,
                },
                layout::PlaneLayout::YUV444([y, u, v]) => match i {
                    0 => *y,
                    1 => *u,
                    2 => *v,
                    _ => *y,
                },
                layout::PlaneLayout::RGB(fmt) => *fmt,
            };

            let hal_descriptor = wgpu::hal::TextureDescriptor {
                label: Some("ffgpu Vulkan Video import"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: vulkan_frames_ctx.nb_layers as u32,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: plane_format,
                usage: wgpu::TextureUses::RESOURCE,
                view_formats: vec![],
                memory_flags: wgpu::hal::MemoryFlags::empty(),
            };

            // Provide a no-op drop callback so that wgpu-hal sets the drop guard and
            // does not call vkDestroyImage on the FFmpeg-owned VkImage.
            let drop_callback: Option<wgpu::hal::DropCallback> =
                Some(Box::new(|| ()));
            let hal_texture = unsafe {
                (*hal_device).texture_from_raw(
                    vk_image,
                    &hal_descriptor,
                    drop_callback,
                    wgpu::hal::vulkan::TextureMemory::External,
                )
            };

            let wgpu_descriptor = wgpu::TextureDescriptor {
                label: Some("ffgpu Vulkan Video import"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: vulkan_frames_ctx.nb_layers as u32,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: plane_format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            };

            let texture = unsafe {
                device.create_texture_from_hal::<wgpu::hal::vulkan::Api>(hal_texture, &wgpu_descriptor)
            };

            // Evict oldest cache entry if at capacity.
            if self.texture_cache.len() >= TEXTURE_CACHE_SIZE {
                self.texture_cache.pop_front();
            }

            self.texture_cache.push_back(CachedTexture {
                vk_image,
                width,
                height,
                texture: texture.clone(),
                identity: texture_format.as_identity(),
            });

            textures.push(texture);
        }

        // Create texture views and a bind group from the imported textures.
        let views: Vec<_> = textures
            .iter()
            .map(|t| t.create_view(&wgpu::TextureViewDescriptor::default()))
            .collect();

        // Build a PlaneLayout of views matching the texture format descriptor.
        let plane_views = match &texture_format.planes {
            layout::PlaneLayout::PackedYUV420(_) => {
                let mut iter = views.into_iter();
                layout::PlaneLayout::PackedYUV420([
                    iter.next().unwrap(),
                    iter.next().unwrap(),
                ])
            }
            layout::PlaneLayout::YUV420(_) => {
                let mut iter = views.into_iter();
                layout::PlaneLayout::YUV420([
                    iter.next().unwrap(),
                    iter.next().unwrap(),
                    iter.next().unwrap(),
                ])
            }
            layout::PlaneLayout::YUV444(_) => {
                let mut iter = views.into_iter();
                layout::PlaneLayout::YUV444([
                    iter.next().unwrap(),
                    iter.next().unwrap(),
                    iter.next().unwrap(),
                ])
            }
            layout::PlaneLayout::RGB(_) => {
                layout::PlaneLayout::RGB(views.into_iter().next().unwrap())
            }
        };

        let frame_descriptor = layout::FrameDescriptor {
            planes: plane_views,
            depth: texture_format.depth,
        };

        self.bg0 = Some(pipeline_cache.bind_frame_textures(
            &frame_descriptor,
            frame.colorspace.into(),
        ));
        self.identity = Some(texture_format.as_identity());

        // Signal the queue again so the next frame can wait. With the queue
        // idle approach above, this is effectively a no-op, but it keeps the
        // placeholder for future semaphore-based synchronization.
        let _ = queue;

        Ok(())
    }

    fn layout_identity(&self) -> Option<layout::FrameDescriptor<()>> {
        self.identity
    }

    fn bind_group(&self) -> Option<&wgpu::BindGroup> {
        self.bg0.as_ref()
    }

    fn name(&self) -> &'static str {
        "Vulkan Video zero-copy"
    }
}
